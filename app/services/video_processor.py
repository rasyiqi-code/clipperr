import cv2
import numpy as np
import os
import json
import math

import clipperr_core
from services.transcription import TranscriptionService
from services.analysis import AnalysisService
from services.face_tracking import FaceTrackingService
from services.thumbnail_service import ThumbnailService
from services.downloader import ModelManager
from config import prefs
from config import (
    WHISPER_MODEL_PATH, LLM_MODEL_ID, OUTPUT_DIR,
    TRANSCRIPTION_LANGUAGE, FACE_TRACKING_SAMPLES,
    FACE_CLUSTER_THRESHOLD, FACE_TEMPORAL_ALPHA,
    CAMERA_CUT_AREA_RATIO, CAMERA_CUT_POSITION_SHIFT, CAMERA_CUT_MIN_SEGMENT,
    FFMPEG_EXE, FFPROBE_EXE,
)
from logger import get_logger

from services.audio_utils import AudioEnergyCache

log = get_logger(__name__)


class VideoProcessor:
    """Orchestrates the full video-clipping pipeline."""

    def __init__(self):
        log.info("Initializing VideoProcessor components")
        self.transcriber = TranscriptionService(model_path=WHISPER_MODEL_PATH)
        self.analyzer = AnalysisService(model_id=LLM_MODEL_ID)
        self.tracker = FaceTrackingService()
        self.model_manager = ModelManager()
        self.global_last_x = 0.5  # Track last known successful face coordinate
        self.audio_cache = AudioEnergyCache()
        self.thumbnail_service = ThumbnailService(self.analyzer)
        log.info("VideoProcessor components initialized")

    # ══════════════════════════════════════════════════
    #  Main pipeline
    # ══════════════════════════════════════════════════
    def process_file(self, video_path, progress_callback=None, cancel_check=None):
        """
        Full pipeline: Metadata → Transcribe → Analyze → Reframe → Render.
        """
        _cancelled = cancel_check or (lambda: False)
        errors: list[str] = []

        log.info("Starting pipeline for %s", video_path)
        self.global_last_x = 0.5 # Reset memory for new video processing mission

        # 1. Metadata
        if progress_callback:
            progress_callback("Extracting metadata...", 5)
        metadata = clipperr_core.extract_metadata(FFPROBE_EXE, video_path)

        # 2. Check Models (Diarization Removed, Face → BlazeFace)
        for m in ["whisper-base", "llm-analysis", "blazeface", "yunet-face"]:
            if not self.model_manager.check_status(m):
                raise FileNotFoundError(f"Model '{m}' is missing. Go to Settings/Models to download it.")

        # 3. Transcription (Diarization Removed)
        if progress_callback:
            progress_callback("Transcribing audio...", 10)
        self.transcriber.load_model()
        segments, info = self.transcriber.transcribe(
            video_path,
            language=TRANSCRIPTION_LANGUAGE,
            progress_callback=progress_callback,
        )

        if _cancelled(): return self._empty_result(metadata)

        # 5. Analyze Moments
        if progress_callback:
            progress_callback("Analyzing viral moments...", 45)
        
        def analysis_progress_wrapper(pct):
            if progress_callback:
                # Scale 0.0-1.0 progress from AnalysisService into the 45% - 49% range
                p_val = 45 + int(pct * 4)
                progress_callback(f"Analyzing viral moments ({int(pct*100)}%)...", p_val)

        viral_clips, analysis_errors = self.analyzer.analyze_transcript(
            segments, 
            progress_callback=analysis_progress_wrapper
        )
        
        # 5.5 Batch Clickbait Generation (Memory Safety: Do this BEFORE rendering loop)
        if viral_clips and prefs.auto_thumbnail:
            if progress_callback:
                progress_callback("Generating AI clickbait titles...", 49)
            
            # Prepare script excerpts for each clip
            scripts = []
            for clip in viral_clips:
                clip_segments = [s for s in segments if s["start"] >= clip["start"] and s["end"] <= clip["end"]]
                scripts.append(" ".join([s["text"] for s in clip_segments]))
            
            # Batch generate in one model-load cycle
            titles = self.analyzer.batch_generate_clickbait(scripts)
            for clip, title in zip(viral_clips, titles):
                clip["clickbait_title"] = title
        
        # 6. Rendering
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        from services.subtitle_service import SubtitleService
        sub_service = SubtitleService()

        processed_clips = []
        total_clips = len(viral_clips)

        for i, clip in enumerate(viral_clips):
            if _cancelled(): break

            step_progress = 50 + int(((i + 1) / max(total_clips, 1)) * 45)
            clip_title = clip.get("title", f"Clip {i+1}")
            try:
                if progress_callback:
                    progress_callback(f"Rendering Clip {i+1}/{total_clips}: {clip_title}", step_progress)

                log.info("Clip %d: Tracking most prominent subject (%.1fs-%.1fs)...", 
                         i + 1, clip["start"], clip["end"])
                
                # Get dynamic panning data (center_x fallback + keyframes for smooth panning)
                center_x, keyframes = self._get_dynamic_focus(
                    video_path, clip["start"], clip["end"],
                    progress_callback=progress_callback
                )
                
                # Serialize keyframes for Rust core
                keyframes_json = json.dumps(keyframes) if keyframes and len(keyframes) >= 2 else None
                
                log.info("Clip %d: Focus X=%.2f, %d keyframes", 
                         i + 1, center_x, len(keyframes) if keyframes else 0)

                output_name = f"clip_{i}_{os.path.basename(video_path)}"
                output_path = os.path.join(OUTPUT_DIR, output_name)
                
                ass_name = f"sub_{i}_{os.path.basename(video_path)}.ass"
                ass_path = os.path.join(OUTPUT_DIR, ass_name)
                sub_service.generate_ass(segments, clip["start"], clip["end"], ass_path)

                # Fetch dynamic settings
                prefs.load()
                wm_json = None
                if prefs.watermark_type == "image" and prefs.watermark_path:
                    wm_json = json.dumps({
                        "type": "image",
                        "path": prefs.watermark_path,
                        "pos": prefs.watermark_pos,
                        "opacity": prefs.watermark_opacity
                    })
                elif prefs.watermark_type == "text" and prefs.watermark_text:
                    wm_json = json.dumps({
                        "type": "text",
                        "text": prefs.watermark_text,
                        "pos": prefs.watermark_pos,
                        "opacity": prefs.watermark_opacity
                    })

                clipperr_core.render_clip(
                    FFMPEG_EXE,
                    video_path, output_path,
                    clip["start"],
                    clip["end"] - clip["start"],
                    center_x,
                    ass_path,
                    keyframes_json,  # Dynamic panning!
                    wm_json,
                )
                
                # Check for AI Thumbnail generation
                thumb_path = None
                if prefs.auto_thumbnail:
                    clip_text = clip.get("description", "Viral Clip")
                    out_thumb_path = os.path.join(OUTPUT_DIR, f"thumb_{i}_{os.path.basename(video_path)}.jpg")
                    thumb_path = self.thumbnail_service.generate_thumbnail(
                        video_path,
                        clip["start"] + 2.0,
                        clip_text,
                        out_thumb_path,
                        center_x,
                        pre_generated_title=clip.get("clickbait_title")
                    )

                result_data = {
                    **clip, 
                    "output_path": output_path, 
                    "speaker": clip.get("speaker", "AI-TRACK") # Generic label for UI
                }
                if thumb_path:
                    result_data["thumbnail_path"] = thumb_path
                    
                processed_clips.append(result_data)
                log.info("Clip %d rendered: %s", i, output_path)

            except Exception as exc:
                log.error("Clip %d failed: %s", i + 1, exc)
                errors.append(str(exc))

        self.tracker.cleanup()
        self.audio_cache.clear()
        if progress_callback: progress_callback("Done!", 100)

        return {
            "clips": processed_clips,
            "metadata": metadata,
            "errors": errors + analysis_errors
        }


    # ══════════════════════════════════════════════════
    #  Dynamic Focus (Keyframe Generator)
    # ══════════════════════════════════════════════════
    # ══════════════════════════════════════════════════
    #  Dynamic Focus (Keyframe Generator with KLT)
    # ══════════════════════════════════════════════════
    def _get_dynamic_focus(self, video_path: str, start: float, end: float, progress_callback=None) -> tuple[float, list[dict]]:
        """
        Generate dynamic panning keyframes for a clip using BlazeFace + KLT Tracker.
        """
        cap = cv2.VideoCapture(
            video_path, 
            cv2.CAP_FFMPEG, 
            [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE]
        )
        if not cap.isOpened():
            return 0.5, []

        duration = end - start
        samples_count = max(2, FACE_TRACKING_SAMPLES)
        step = duration / (samples_count - 1) if samples_count > 1 else duration
        
        frame_samples = []
        prev_frame_gray = None
        prev_faces = []
        
        try:
            # ── Phase 1: Robust sampling with KLT ──
            for i in range(samples_count):
                ts = start + (i * step)
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                success, frame = cap.read()
                if not success: continue
                
                if progress_callback:
                    p_val = 50 + int((i / samples_count) * 10)
                    progress_callback(f"Scanning frame {i+1}/{samples_count} (Profile Recovery Mode)...", p_val)

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces, scene_info = self.tracker.detect_with_fallback(frame)
                
                # KLT Fallback: If detector fails, track points from previous successful frame
                if not faces and prev_faces and prev_frame_gray is not None:
                    tracked_faces = self._track_with_klt(prev_frame_gray, frame_gray, prev_faces)
                    if tracked_faces:
                        log.debug("[KLT] Bridged detection gap at t=%.1f", ts)
                        faces = tracked_faces

                frame_samples.append({
                    "ts": ts,
                    "faces": faces,
                    "scene": scene_info,
                })
                
                if faces:
                    prev_faces = faces
                    prev_frame_gray = frame_gray
            
            if not frame_samples:
                return 0.5, []
            
            # ── Phase 2: Camera cuts & Sub-segments ──
            cut_indices = self._detect_camera_cuts(frame_samples)
            sub_segments = self._split_at_cuts(frame_samples, cut_indices)
            
            # ── Phase 3: Analyze each sub-segment ──
            segment_results = []
            for sub_frames in sub_segments:
                if not sub_frames: continue
                
                scene_types = [f["scene"]["type"] for f in sub_frames]
                dominant_scene = max(set(scene_types), key=scene_types.count)
                params = sub_frames[0]["scene"]["adaptive_params"]
                
                # Calculate mouth activity (variation in mouth corner distance or relative Y)
                mouth_activity = self._calculate_mouth_activity(sub_frames)
                
                clusters = self._cluster_faces_2d(sub_frames, params.get("cluster_threshold", 0.07))
                if not clusters: continue
                
                scored = [self._score_cluster(c, mouth_activity) for c in clusters]
                scored.sort(key=lambda m: m["importance"], reverse=True)
                best = scored[0]
                
                # Apply Speaker Lock: If previous segment had a speaker at a similar X, 
                # give them a performance boost to prevent jittery jumps.
                if segment_results:
                    last_x = segment_results[-1]["center_x"]
                    if abs(best["avg_cx"] - last_x) < 0.15:
                        best["importance"] *= 1.2 # Inertia bonus
                
                segment_results.append({
                    "center_x": best["avg_cx"],
                    "importance": best["importance"],
                    "duration": sub_frames[-1]["ts"] - sub_frames[0]["ts"],
                    "rel_start": sub_frames[0]["ts"] - start,
                    "rel_end": sub_frames[-1]["ts"] - start,
                })
            
            if not segment_results:
                log.warning("[FOCUS] No face clusters found in clip. Falling back to last known position X=%.2f", self.global_last_x)
                return self.global_last_x, []
            
            # ── Phase 4: Final Winner & Keyframes ──
            segment_results.sort(key=lambda r: r["importance"] * math.sqrt(r["duration"]), reverse=True)
            static_winner = segment_results[0]
            static_cx = static_winner["center_x"]
            self.global_last_x = static_cx
            
            if len(segment_results) <= 1:
                return float(static_cx), []
            
            # Generate smooth transitions
            segment_results.sort(key=lambda r: r["rel_start"])
            keyframes = []
            TRANS_DUR = 0.5
            
            for i, seg in enumerate(segment_results):
                cx = seg["center_x"]
                t_start = seg["rel_start"]
                
                if i == 0:
                    keyframes.append({"t": 0.0, "x": round(cx, 4)})
                else:
                    prev_x = keyframes[-1]["x"]
                    trans_s = max(t_start - TRANS_DUR/2, keyframes[-1]["t"] + 0.1)
                    trans_e = min(t_start + TRANS_DUR/2, duration)
                    keyframes.append({"t": round(trans_s, 3), "x": prev_x})
                    keyframes.append({"t": round(trans_e, 3), "x": round(cx, 4)})
                    
                if i == len(segment_results) - 1:
                    keyframes.append({"t": round(duration, 3), "x": round(cx, 4)})
            
            # Final Temporal Smoothing
            smoothed_x = self._apply_temporal_smoothing([kf["x"] for kf in keyframes])
            for kf, sx in zip(keyframes, smoothed_x):
                kf["x"] = round(sx, 4)
                
            return float(static_cx), keyframes
            
        finally:
            cap.release()

    def _track_with_klt(self, prev_gray, curr_gray, prev_faces):
        """Bridge detection gaps using Optical Flow."""
        points = []
        for f in prev_faces:
            bbox = f["bbox"]
            points.append([bbox["origin_x"] + bbox["width"]/2, bbox["origin_y"] + bbox["height"]/2])
        
        if not points: return []
        
        p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
        
        h, w = curr_gray.shape
        new_faces = []
        for i, (new_p, status) in enumerate(zip(p1, st)):
            if status == 1:
                nx, ny = new_p.ravel()
                old_f = prev_faces[i]
                new_faces.append({
                    "center_x": nx / w,
                    "center_y": ny / h,
                    "score": old_f["score"] * 0.9, # Decaying confidence
                    "area": old_f["area"],
                    "confirmed": False,
                    "bbox": old_f["bbox"] # Keep old bbox size, just move center
                })
        return new_faces

    # ══════════════════════════════════════════════════
    #  Robust Face Center (Single-point, delegates to _get_dynamic_focus)
    # ══════════════════════════════════════════════════
    def _get_robust_face_center(self, video_path: str, start: float, end: float) -> float:
        """Thin wrapper returning only center_x."""
        center_x, _ = self._get_dynamic_focus(video_path, start, end)
        return center_x

    # ══════════════════════════════════════════════════
    #  Camera Cut Detection
    # ══════════════════════════════════════════════════
    def _detect_camera_cuts(self, frame_samples: list[dict]) -> list[int]:
        """
        Detect camera cuts between consecutive frame samples.
        
        A cut is detected when:
        - Face count changes significantly (Δ >= 2)
        - Average face area jumps by CAMERA_CUT_AREA_RATIO (e.g., 2x)
        - All face positions shift by > CAMERA_CUT_POSITION_SHIFT simultaneously
        
        Returns list of indices where cuts occur (index = frame AFTER the cut).
        """
        cuts = []
        
        for i in range(1, len(frame_samples)):
            prev = frame_samples[i - 1]
            curr = frame_samples[i]
            
            prev_faces = prev["faces"]
            curr_faces = curr["faces"]
            
            # Check minimum segment duration before allowing another cut
            if cuts:
                last_cut_ts = frame_samples[cuts[-1]]["ts"]
                if curr["ts"] - last_cut_ts < CAMERA_CUT_MIN_SEGMENT:
                    continue
            
            is_cut = False
            
            # ── Check 1: Face count change ──
            prev_count = len(prev_faces)
            curr_count = len(curr_faces)
            if abs(prev_count - curr_count) >= 2:
                is_cut = True
                log.debug("[CUT] Face count change %d→%d at t=%.1f", 
                          prev_count, curr_count, curr["ts"])
            
            # ── Check 2: Area ratio jump ──
            if not is_cut and prev_faces and curr_faces:
                prev_avg_area = sum(f["area"] for f in prev_faces) / len(prev_faces)
                curr_avg_area = sum(f["area"] for f in curr_faces) / len(curr_faces)
                
                if prev_avg_area > 0 and curr_avg_area > 0:
                    ratio = max(prev_avg_area, curr_avg_area) / min(prev_avg_area, curr_avg_area)
                    if ratio >= CAMERA_CUT_AREA_RATIO:
                        is_cut = True
                        log.debug("[CUT] Area ratio %.1fx at t=%.1f", ratio, curr["ts"])
            
            # ── Check 3: Mass position shift ──
            if not is_cut and prev_faces and curr_faces:
                prev_positions = sorted([f["center_x"] for f in prev_faces])
                curr_positions = sorted([f["center_x"] for f in curr_faces])
                
                # Compare overlapping positions (handles slight face count changes)
                min_len = min(len(prev_positions), len(curr_positions))
                if min_len > 0:
                    shifts = [abs(p - c) for p, c in zip(prev_positions[:min_len], curr_positions[:min_len])]
                    if all(s > CAMERA_CUT_POSITION_SHIFT for s in shifts):
                        is_cut = True
                        log.debug("[CUT] Mass position shift at t=%.1f", curr["ts"])
            
            if is_cut:
                cuts.append(i)
        
        return cuts

    def _split_at_cuts(self, frame_samples: list[dict], cut_indices: list[int]) -> list[list[dict]]:
        """Split frame samples into sub-segments at camera cut points."""
        if not cut_indices:
            return [frame_samples]
        
        segments = []
        prev_idx = 0
        for cut_idx in cut_indices:
            if cut_idx > prev_idx:
                segments.append(frame_samples[prev_idx:cut_idx])
            prev_idx = cut_idx
        
        # Add the remaining segment
        if prev_idx < len(frame_samples):
            segments.append(frame_samples[prev_idx:])
        
        return [s for s in segments if s]  # Filter empty segments

    # ══════════════════════════════════════════════════
    #  2D Spatial Clustering
    # ══════════════════════════════════════════════════
    def _cluster_faces_2d(self, sub_frames: list[dict], threshold: float) -> list[list[dict]]:
        """
        Cluster faces using 2D Euclidean distance (center_x, center_y).
        More accurate than X-only clustering — prevents merging faces at same X but different Y.
        """
        clusters: list[list[dict]] = []
        
        for frame_data in sub_frames:
            for face in frame_data["faces"]:
                cx = face["center_x"]
                cy = face.get("center_y", 0.3)
                
                found_cluster = False
                best_dist = float("inf")
                best_cluster_idx = -1
                
                for idx, cluster in enumerate(clusters):
                    avg_cx = sum(f["cx"] for f in cluster) / len(cluster)
                    avg_cy = sum(f["cy"] for f in cluster) / len(cluster)
                    
                    # 2D Euclidean distance (weight Y less since faces vary more vertically)
                    dist = math.sqrt((cx - avg_cx) ** 2 + ((cy - avg_cy) * 0.5) ** 2)
                    
                    if dist < threshold and dist < best_dist:
                        best_dist = dist
                        best_cluster_idx = idx
                
                entry = {
                    "cx": cx, 
                    "cy": cy, 
                    "score": face["score"], 
                    "area": face["area"],
                    "mouth": face.get("mouth_score", 0.0), 
                    "audio": face.get("audio_energy", 0.0),
                    "confirmed": face.get("has_landmarks", False),
                    "speaking": face.get("is_speaking", False),
                    "critical_lm": face.get("critical_landmarks", 0),
                }
                
                if best_cluster_idx >= 0:
                    clusters[best_cluster_idx].append(entry)
                else:
                    clusters.append([entry])
        
        return clusters

    # ══════════════════════════════════════════════════
    #  Cluster Scoring (ASD V8)
    # ══════════════════════════════════════════════════
    def _calculate_mouth_activity(self, sub_frames: list[dict]) -> dict:
        """
        Estimate vertical mouth variation per cluster area to find the active speaker.
        """
        activity = {} # Map cx_bucket -> movement score
        for frame in sub_frames:
            for face in frame["faces"]:
                cx = round(face["center_x"], 2)
                l = face.get("landmarks")
                if l and "mouth_r" in l and "mouth_l" in l:
                    # Use relative distance or movement of corners
                    # For profiles, we look at the deviation from nose
                    activity[cx] = activity.get(cx, 0) + 1
        return activity

    def _score_cluster(self, cluster: list[dict], mouth_activity: dict) -> dict:
        """
        Score a spatial cluster for visual importance.
        """
        count = len(cluster)
        avg_score = sum(f["score"] for f in cluster) / count
        avg_area = sum(f["area"] for f in cluster) / count
        avg_cx = sum(f["cx"] for f in cluster) / count
        
        # Central bias: prefer subjects near the center
        dist_from_center = abs(avg_cx - 0.5)
        central_bonus = max(0, 1.0 - (dist_from_center * 2))

        # Mouth Activity Bonus (Simplified Active Speaker Detection)
        # Check if this cluster's position matches any active mouth zones
        active_bonus = 0
        for active_cx, val in mouth_activity.items():
            if abs(avg_cx - active_cx) < 0.05:
                active_bonus = min(20, val * 2)
                break
        
        importance = (avg_area * 60) + (avg_score * 10) + (central_bonus * 5) + active_bonus
        
        return {
            "importance": importance,
            "avg_cx": avg_cx,
        }

    # ══════════════════════════════════════════════════
    #  Temporal Smoothing
    # ══════════════════════════════════════════════════
    @staticmethod
    def _apply_temporal_smoothing(positions: list[float], alpha: float = FACE_TEMPORAL_ALPHA) -> list[float]:
        """
        Exponential Moving Average to reduce position jitter.
        Alpha 0.3 means 70% previous position, 30% new measurement.
        """
        if not positions:
            return []
        smoothed = [positions[0]]
        for p in positions[1:]:
            smoothed.append(alpha * p + (1 - alpha) * smoothed[-1])
        return smoothed

    # ══════════════════════════════════════════════════
    #  Memory Fallback
    # ══════════════════════════════════════════════════
    def _get_memory_fallback(self) -> float:
        """Retrieve last known successful face position."""
        return self.global_last_x

    @staticmethod
    def _empty_result(metadata=None):
        return {"metadata": metadata, "clips": [], "segments": [], "errors": []}
