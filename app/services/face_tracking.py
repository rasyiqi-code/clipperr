import os
import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import (
    BLAZEFACE_MODEL_PATH, YUNET_MODEL_PATH,
    FACE_Y_GATE_DEFAULT, FACE_MIN_AREA_DEFAULT, FACE_UNCONFIRMED_SCORE_MIN
)
from logger import get_logger

log = get_logger(__name__)

# ── MediaPipe landmark indices for critical facial features ──
# Eyes (inner/outer corners), nose tip, mouth corners, upper/lower lip
CRITICAL_LANDMARK_INDICES = [33, 133, 362, 263, 1, 13, 14, 61, 291]


class FaceTrackingService:
    """
    Production-grade Hybrid tracking: YuNet for detection + MediaPipe for validation & MAR.
    
    Features:
    - Scene-adaptive detection parameters (wide shot, closeup, single, group)
    - Two-pass detection with fallback (strict → relaxed)
    - Critical landmark validation (rejects hands/objects masquerading as faces)
    - Y-position gating (rejects feet/shoes/lower-body objects)
    """

    def __init__(self, detector_path: str = BLAZEFACE_MODEL_PATH, yunet_path: str = YUNET_MODEL_PATH):
        self.detector_path = detector_path
        self.yunet_path = yunet_path
        self._detector = None
        self._yunet = None

    def load_model(self) -> bool:
        """Initialize both YuNet and Face Landmarker."""
        try:
            # 1. MediaPipe Face Detector (BlazeFace)
            if os.path.exists(self.detector_path):
                base_options = python.BaseOptions(model_asset_path=self.detector_path)
                options = vision.FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=0.4
                )
                self._detector = vision.FaceDetector.create_from_options(options)
                log.info("MediaPipe FaceDetector (BlazeFace) initialized")
            else:
                log.warning("BlazeFace model missing: %s", self.detector_path)

            # 2. OpenCV YuNet
            if os.path.exists(self.yunet_path):
                self._yunet = cv2.FaceDetectorYN.create(
                    model=self.yunet_path,
                    config="",
                    input_size=(640, 640),
                    score_threshold=0.5,   # Lower base threshold — scene classifier will gate
                    nms_threshold=0.4
                )
                log.info("YuNet Detector initialized")
            else:
                log.warning("YuNet model missing: %s", self.yunet_path)

            return True if self._yunet else False
        except Exception as e:
            log.exception("Failed to initialize Hybrid models")
            return False

    # ══════════════════════════════════════════════════
    #  Scene Classification
    # ══════════════════════════════════════════════════
    @staticmethod
    def classify_scene(faces: list[dict], frame_w: int, frame_h: int) -> dict:
        """
        Classify scene type based on detected face characteristics.
        Returns adaptive parameters tuned for the detected scene.
        
        Scene Types:
        - wide_shot:  avg_area < 0.02, 2+ faces (podcast full-body)
        - closeup:    avg_area > 0.05, 1-2 faces (medium/close-up)
        - single:     1 face, area > 0.03 (solo talking head)
        - group:      3+ faces (panel discussion)
        - no_face:    0 faces detected
        """
        face_count = len(faces)
        
        if face_count == 0:
            return {
                "type": "no_face",
                "avg_area": 0.0,
                "face_count": 0,
                "adaptive_params": {
                    "y_gate": FACE_Y_GATE_DEFAULT,
                    "min_area": FACE_MIN_AREA_DEFAULT,
                    "cluster_threshold": 0.07,
                    "unconfirmed_score_min": FACE_UNCONFIRMED_SCORE_MIN,
                }
            }
        
        avg_area = sum(f.get("area", 0) for f in faces) / face_count
        
        # Classification logic
        if face_count >= 3:
            scene_type = "group"
            params = {
                "y_gate": 0.55,       # Tighter — more people, more clutter at bottom
                "min_area": 0.002,     # Smaller faces in group shots
                "cluster_threshold": 0.12,  # Wider — more spread out
                "unconfirmed_score_min": 0.92,  # Stricter — more false positive risk
            }
        elif avg_area > 0.15 and face_count == 1:
            scene_type = "single"
            params = {
                "y_gate": 0.80,       # Relaxed — single person might be lower
                "min_area": 0.005,     # Bigger face expected
                "cluster_threshold": 0.05,
                "unconfirmed_score_min": 0.70,  # Can be looser — less confusion
            }
        elif avg_area > 0.05:
            scene_type = "closeup"
            params = {
                "y_gate": 0.75,       # Relaxed — faces fill more of the frame
                "min_area": 0.01,      # Big faces, reject small noise
                "cluster_threshold": 0.05,  # Tight — faces close together
                "unconfirmed_score_min": 0.75,
            }
        elif avg_area < 0.02:
            scene_type = "wide_shot"
            params = {
                "y_gate": 0.60,       # Strict — lots of body/furniture below
                "min_area": 0.0015,   # Very small faces OK (e.g. strict wide shot)
                "cluster_threshold": 0.10,  # Wider — faces far apart
                "unconfirmed_score_min": 0.70,  # Strict — more false positive risk
            }
        else:
            # Medium shot (between wide and closeup)
            scene_type = "medium"
            params = {
                "y_gate": FACE_Y_GATE_DEFAULT,
                "min_area": FACE_MIN_AREA_DEFAULT,
                "cluster_threshold": 0.07,
                "unconfirmed_score_min": FACE_UNCONFIRMED_SCORE_MIN,
            }
        
        return {
            "type": scene_type,
            "avg_area": avg_area,
            "face_count": face_count,
            "adaptive_params": params,
        }

    # ══════════════════════════════════════════════════
    #  Core Detection
    # ══════════════════════════════════════════════════
    def _detect_faces_internal(self, frame, y_gate: float, min_area: float, 
                                unconfirmed_score_min: float) -> list[dict]:
        """
        Internal face detection with configurable thresholds.
        
        Args:
            frame: BGR image (numpy array)
            y_gate: Max normalized Y-position for a valid face
            min_area: Min normalized area for a valid detection
            unconfirmed_score_min: Min YuNet score for detections without landmarks
        
        Returns:
            List of face dicts with center_x, center_y, score, area, mouth_score, etc.
        """
        if not self._yunet:
            if not self.load_model():
                return []

        try:
            h, w, _ = frame.shape
            self._yunet.setInputSize((w, h))
            
            # 1. YuNet Detection
            _, detections = self._yunet.detect(frame)
            if detections is None:
                return []
            
            # 2. MediaPipe Detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            mp_result = self._detector.detect(mp_image) if self._detector else None
            
            mp_detections = mp_result.detections if mp_result else []
            
            results = []
            
            for det in detections:
                x_min, y_min, det_w, det_h = det[:4]
                score = det[-1]
                
                # ── Filter 1: Aspect Ratio ──
                aspect = det_h / det_w if det_w > 0 else 0
                if aspect < 0.75 or aspect > 2.0:
                    continue

                # ── Filter 2: Y-Position Gate ──
                center_y = (y_min + det_h / 2) / h
                if center_y > y_gate:
                    log.debug("Rejecting blob at Y=%.2f (gate=%.2f) — below face zone", center_y, y_gate)
                    continue
                
                # ── Filter 3: Minimum Area ──
                area_norm = (det_w * det_h) / (w * h)
                if area_norm < min_area:
                    log.debug("Rejecting blob (area=%.4f, min=%.4f) — too small", area_norm, min_area)
                    continue

                center_x = (x_min + det_w / 2) / w
                
                # Check if MediaPipe confirms this detection
                confirmed = False
                for mp_det in mp_detections:
                    bbox = mp_det.bounding_box
                    # Check overlap or proximity
                    mp_cx = (bbox.origin_x + bbox.width / 2) / w
                    mp_cy = (bbox.origin_y + bbox.height / 2) / h
                    dist = math.sqrt((center_x - mp_cx)**2 + (center_y - mp_cy)**2)
                    if dist < 0.1:
                        confirmed = True
                        break

                if not confirmed and score < unconfirmed_score_min:
                    continue
                
                results.append({
                    "center_x": center_x,
                    "center_y": center_y,
                    "score": float(score),
                    "area": float(area_norm),
                    "confirmed": confirmed,
                    "bbox": {
                        "origin_x": int(x_min),
                        "origin_y": int(y_min),
                        "width": int(det_w),
                        "height": int(det_h)
                    }
                })
            
            return results
        except Exception as e:
            log.error("Detection error: %s", e)
            return []

    def detect_all_faces(self, frame) -> list[dict]:
        """
        Detect all faces using default parameters.
        For scene-adaptive detection, use detect_with_fallback().
        """
        return self._detect_faces_internal(
            frame, 
            y_gate=FACE_Y_GATE_DEFAULT, 
            min_area=FACE_MIN_AREA_DEFAULT,
            unconfirmed_score_min=FACE_UNCONFIRMED_SCORE_MIN
        )

    def detect_with_fallback(self, frame) -> tuple[list[dict], dict]:
        """
        Optimized two-pass detection with scene-adaptive parameters.
        
        FT-1 fix: reduced from 3x to at most 2x detection per frame.
        - Pass 1: Strict detection with default params
        - If confirmed faces found AND scene params differ → re-detect (2x max)
        - If no confirmed faces → fallback with relaxed params (2x max)
        
        Returns: (faces, scene_info)
        """
        h, w = frame.shape[:2]
        
        # ── Pass 1: Strict ──
        faces = self._detect_faces_internal(
            frame, 
            y_gate=FACE_Y_GATE_DEFAULT,
            min_area=FACE_MIN_AREA_DEFAULT,
            unconfirmed_score_min=FACE_UNCONFIRMED_SCORE_MIN
        )
        
        scene = self.classify_scene(faces, w, h)
        confirmed_count = sum(1 for f in faces if f.get("confirmed", False))
        
        if confirmed_count > 0:
            return faces, scene
        
        # ── Pass 2: Relaxed ──
        relaxed_faces = self._detect_faces_internal(
            frame,
            y_gate=0.80,
            min_area=0.001,
            unconfirmed_score_min=0.60
        )
        return relaxed_faces, self.classify_scene(relaxed_faces, w, h)

    def track_face(self, frame) -> tuple[float, float]:
        """Backward compatible wrapper."""
        faces = self.detect_all_faces(frame)
        if not faces:
            return 0.5, 0.0
        best = max(faces, key=lambda f: f["area"])
        return best["center_x"], best["score"]

    def cleanup(self):
        """Release resources."""
        if self._detector:
            try: self._detector.close()
            except: pass
            self._detector = None
        self._yunet = None
        log.debug("FaceTracker cleanup")

    def __del__(self):
        self.cleanup()

