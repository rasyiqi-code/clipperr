import json
import requests
import math
import re
from config import (
    LLM_MAX_TOKENS, LLM_TOP_K, LLM_TEMPERATURE, MAX_VIRAL_CLIPS, 
    LLM_MODEL_ID, prefs, ANALYSIS_CHUNK_DURATION, ANALYSIS_CHUNK_OVERLAP
)
from logger import get_logger

# Silence noisy external libraries
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

log = get_logger(__name__)


class AnalysisService:
    def __init__(self, model_id: str = LLM_MODEL_ID):
        self.model_id = model_id
        self.llm = None
        self.available = False

    def load_model(self):
        """
        Try to load LLM via Hugging Face Transformers.
        Falls back to heuristic analysis if unavailable.
        """
        try:
            import torch
            from transformers import pipeline

            from config import LLM_MODEL_PATH
            log.info("Initializing Hugging Face pipeline from local path: %s...", LLM_MODEL_PATH)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

            self.llm = pipeline(
                "text-generation",
                model=LLM_MODEL_PATH,
                device=device,
                torch_dtype=torch_dtype,
                model_kwargs={"low_cpu_mem_usage": True},
            )
            # Force remove default max_length to stop the warning about conflicting with max_new_tokens
            if hasattr(self.llm, "model") and hasattr(self.llm.model, "generation_config"):
                self.llm.model.generation_config.max_length = None
                
            self.available = True
            log.info("LLM pipeline loaded successfully on %s", device)
        except Exception as exc:
            log.warning("LLM not available (%s) — using heuristic analysis", exc)

    def analyze_transcript(self, transcript_segments: list[dict], progress_callback=None) -> tuple[list[dict], list[str]]:
        """
        Analyze transcript to find viral moments.
        Large transcripts are split into chunks to avoid context window limits.
        """
        if not transcript_segments:
            return [], ["No transcript segments provided."]
        
        total_duration = transcript_segments[-1]["end"]
        
        # Threshold: if > 1.2x chunk size, use chunking
        if total_duration > (ANALYSIS_CHUNK_DURATION * 1.2):
            return self._chunked_analyze(transcript_segments, progress_callback)
        
        # Single-pass analysis for shorter videos
        clips = self._single_pass_analyze(transcript_segments)
        if progress_callback: progress_callback(1.0)
        return clips, []

    def _single_pass_analyze(self, transcript_segments: list[dict]) -> list[dict]:
        """Standard single-pass AI analysis."""
        try:
            prefs.load()
            if prefs.llm_provider == "api":
                return self._api_analyze(transcript_segments)
            
            if not self.available:
                self.load_model()

            if self.llm and self.available:
                return self._llm_analyze(transcript_segments)
            
            # Graceful degradation: use heuristic instead of crashing
            log.warning("LLM unavailable — falling back to heuristic analysis")
            return self._heuristic_analyze(transcript_segments)
        finally:
            self.unload_model()

    def _chunked_analyze(self, transcript_segments: list[dict], progress_callback=None) -> tuple[list[dict], list[str]]:
        """Coordinate multi-pass chunked analysis."""
        chunks = self._split_into_chunks(transcript_segments)
        log.info("Processing long video in %d chunks...", len(chunks))
        
        all_clips = []
        errors = []
        for i, chunk_segments in enumerate(chunks):
            log.info("Analyzing chunk %d/%d...", i + 1, len(chunks))
            try:
                # We use _single_pass_analyze but we DON'T unload the model every time
                # to save loading time during the sequence.
                # However, for APIs it doesn't matter. 
                # For Local, we'll keep it loaded until the end.
                if prefs.llm_provider == "api":
                    chunk_clips = self._api_analyze(chunk_segments)
                else:
                    if not self.available: self.load_model()
                    if self.llm and self.available:
                        chunk_clips = self._llm_analyze(chunk_segments)
                    else:
                        # Heuristic fallback per-chunk
                        chunk_clips = self._heuristic_analyze(chunk_segments)
                
                all_clips.extend(chunk_clips)
                
                if progress_callback:
                    # Report progress proportional to chunks processed (0.0 to 1.0)
                    progress_callback((i + 1) / len(chunks))

            except Exception as e:
                msg = str(e)
                log.error("Error analyzing chunk %d: %s", i + 1, msg)
                errors.append(f"AI Analysis (Chunk {i+1}): {msg}")
        
        if not prefs.llm_provider == "api":
            self.unload_model()

        # Merge and limit
        processed_clips = self._post_process_clips(all_clips)
        return processed_clips, errors

    def _split_into_chunks(self, segments: list[dict]) -> list[list[dict]]:
        """Split segments into timed chunks with overlaps."""
        duration = segments[-1]["end"]
        chunks = []
        
        start_ts = 0.0
        while start_ts < duration:
            end_ts = start_ts + ANALYSIS_CHUNK_DURATION
            
            # Extract segments within this window
            chunk_segs = [
                s for s in segments 
                if (s["start"] >= start_ts and s["start"] < end_ts) or
                   (s["end"] > start_ts and s["end"] <= end_ts)
            ]
            
            if chunk_segs:
                chunks.append(chunk_segs)
            
            # Slide window forward by duration minus overlap
            start_ts += (ANALYSIS_CHUNK_DURATION - ANALYSIS_CHUNK_OVERLAP)
            
        return chunks

    def _post_process_clips(self, clips: list[dict]) -> list[dict]:
        """Deduplicate overlaps and limit to top results."""
        if not clips: return []
        
        # 1. Deduplicate: if two clips start within 2 seconds, keep the longer/higher scored
        clips.sort(key=lambda x: x["start"])
        unique_clips = []
        
        for c in clips:
            is_dup = False
            for u in unique_clips:
                if abs(c["start"] - u["start"]) < 5.0: # 5s threshold for "same moment"
                    is_dup = True
                    # Keep the one with better metadata if possible
                    if len(c.get("description", "")) > len(u.get("description", "")):
                        u.update(c)
                    break
            if not is_dup:
                unique_clips.append(c)
        
        # 2. Sort by scores if available, otherwise just count
        unique_clips.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        return unique_clips[:prefs.max_viral_clips]

    def unload_model(self):
        """Purge model from memory to free up resources for video processing."""
        if not self.llm:
            return

        import gc
        import torch
        
        log.info("Unloading LLM model to free memory...")
        self.llm = None
        self.available = False
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("LLM unloaded successfully.")

    # ══════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════
    def _sanitize_transcript(self, text: str) -> str:
        """Sanitize transcript to prevent prompt injection and formatting issues."""
        # Remove character that might look like prompt instructions or break JSON
        text = text.replace('"', "'")  # Replace double quotes with single to avoid JSON nesting issues
        text = re.sub(r'([\\`])', r'\\\1', text) # Escape backslashes and backticks
        return text.strip()

    def _repair_json(self, json_str: str) -> str:
        """Robustly repair common LLM JSON mistakes."""
        # 1. Basic cleaning
        json_str = json_str.strip()
        if not json_str:
            return "[]"
        
        # 2. Support markdown fences
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
            
        # 3. Robust extraction: find first '[' and last ']' (our schema is always a list)
        start_idx = json_str.find("[")
        end_idx = json_str.rfind("]")
        if start_idx != -1 and end_idx != -1:
            json_str = json_str[start_idx : end_idx + 1]
        elif start_idx != -1:
            # Truncated JSON: Starts but never ends
            json_str = json_str[start_idx:] + "]"
        else:
            # No JSON brackets found, return empty list to avoid crashing
            return "[]"

        # 4. Remove single-line comments
        json_str = re.sub(r'//.*', '', json_str)
        
        # 5. Support unquoted keys (e.g., { start: 10 } -> { "start": 10 })
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', json_str)
        
        # 6. Support single quotes (only if clearly used as delimiters to avoid breaking words like "don't")
        json_str = re.sub(r"\'(\s*[:}\],])", r'"\1', json_str) # Trailing single quote
        json_str = re.sub(r"([{\[,:\s])\'", r'\1"', json_str) # Leading single quote

        # 7. Add missing commas between key-value pairs OR between objects
        # Pattern 1: value followed by space and then "key":
        json_str = re.sub(r'("|\d|true|false|null)\s+(")', r'\1, \2', json_str)
        # Pattern 2: object end followed by object start: } { -> }, {
        json_str = json_str.replace("} {", "}, {")
        
        # 8. Handle trailing commas which break strict json.loads
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)

        # 9. Final safety: remove invalid escapes
        json_str = json_str.replace("\\'", "'")
        
        # 10. Recursive repair for truncated objects inside list
        # If it ends with "...", try to close it
        if json_str.endswith("..."):
            json_str = json_str[:-3].strip()
            if not json_str.endswith("}"): json_str += "}"
            if not json_str.endswith("]"): json_str += "]"

        return json_str

    # ══════════════════════════════════════════════════
    #  LLM analysis
    # ══════════════════════════════════════════════════
    def _llm_analyze(self, transcript_segments: list[dict]) -> list[dict]:
        full_transcript = [
            f"[{s['start']:.2f}-{s['end']:.2f}] {self._sanitize_transcript(s['text'])}"
            for s in transcript_segments
        ]
        context_text = "\n".join(full_transcript)

        prompt = (
            f"Analyze this video transcript and identify ONLY the most impactful 'viral moments' (maximum 5 high-quality segments).\n"
            "A viral moment is a segment with a strong hook, a deep insight, or a complete self-contained story.\n"
            "- CRITICAL: For podcasts, each clip MUST have a natural duration of 45-90 seconds. Do NOT cut early.\n"
            "- EACH clip MUST NOT exceed 180 seconds.\n"
            "- Ensure the clip starts with the setup (Hook) and ends only after the point is fully made.\n"
            'Return ONLY JSON in this format: [{ "start": 12.5, "end": 75.0, "title": "Hook Name", "description": "...", "hashtags": "#tag1", "explanation": "..." }]\n\n'
            f"Transcript:\n{context_text}"
        )
        
        messages = [{"role": "user", "content": prompt}]

        try:
            # Safety check
            if not self.llm:
                raise Exception("LLM model is not loaded.")

            import torch
            # Optimization: Use all available CPU threads for torch if on CPU
            if not torch.cuda.is_available():
                import multiprocessing
                torch.set_num_threads(multiprocessing.cpu_count())

            # Specific call to avoid transformers warning about conflicting max_length vs max_new_tokens
            # We use return_full_text=False to ONLY get the model's new generation
            outputs = self.llm(
                messages,
                max_new_tokens=prefs.llm_max_tokens,
                do_sample=True,
                temperature=prefs.llm_temperature,
                top_k=prefs.llm_top_k,
                pad_token_id=self.llm.tokenizer.eos_token_id if hasattr(self.llm, 'tokenizer') else None,
                return_full_text=False
            )
            
            # Extract response text
            # Depending on model/version, response could be in different keys
            raw_gen = outputs[0]
            if "generated_text" in raw_gen:
                response = raw_gen["generated_text"]
            else:
                response = str(raw_gen)
            
            # Fallback if text-generation returns a list of messages instead of a string
            if isinstance(response, list) and len(response) > 0:
                response = response[-1].get("content", "")
            
            if not response or not response.strip():
                log.warning("AI returned empty generation. Falling back to heuristic.")
                return self._heuristic_analyze(transcript_segments)

            json_str = self._repair_json(response)
            log.debug("Repaired LLM JSON: %s", json_str)

            try:
                clips = json.loads(json_str)
            except json.JSONDecodeError:
                # AN-2 Fallback: If JSON fails, try to parse it as a conversational list
                log.info("JSON parsing failed, attempting conversational list recovery...")
                clips = self._try_parse_conversational_list(response)
                
                if not clips:
                    # Very basic attempt to fix unescaped quotes in middle of strings
                    try:
                        json_str_fixed = re.sub(r'":\s*"(.*)"\s*([,}])', 
                                                lambda m: '": "' + m.group(1).replace('"', '\\"') + '"' + m.group(2), 
                                                json_str)
                        clips = json.loads(json_str_fixed)
                    except:
                        log.error("Failed to parse LLM JSON. Raw snippet: %s...", response[:200])
                        log.warning("Falling back to heuristic analysis due to JSON failure.")
                        return self._heuristic_analyze(transcript_segments)

            # Normalize keys to ensure UI fields exist
            if not isinstance(clips, list):
                if isinstance(clips, dict): clips = [clips]
                else: raise Exception("AI returned non-list format.")

            normalized_clips = []
            for c in clips:
                if not isinstance(c, dict):
                    continue
                # Map common AI variations back to our schema
                norm = {
                    "start": self._parse_timestamp(c.get("start")),
                    "end": self._parse_timestamp(c.get("end")),
                    "title": c.get("title") or c.get("heading") or c.get("name") or "Viral Clip",
                    "description": c.get("description") or c.get("explanation") or "No description provided.",
                    "hashtags": c.get("hashtags") or "#viral #ai #shorts",
                    "explanation": c.get("explanation") or c.get("description") or ""
                }
                normalized_clips.append(norm)

            log.info("LLM analysis found %d clips", len(normalized_clips))
            return normalized_clips

        except Exception as exc:
            log.error("LLM inference error: %s", exc)
            raise Exception(f"Local AI Inference error: {exc}")

    def _try_parse_conversational_list(self, text: str) -> list[dict]:
        """
        Fallback parser for cases where the LLM returns a conversational list
        instead of JSON.
        Matches patterns like: "1. [10.5-20.3] - Interesting moment"
        """
        clips = []
        # Pattern matches: 
        # 1. Start bracket '['
        # 2. Start time (float or HH:MM:SS)
        # 3. Separator '-' or 'to'
        # 4. End time
        # 5. End bracket ']'
        # 6. Optional separator ': ' or ' - '
        # 7. The description/title
        pattern = r'\[([\d:.]+)\s*(?:-|to)\s*([\d:.]+)\][ \-:]*(.*)'
        
        lines = text.split('\n')
        current_clip = None
        
        for line in lines:
            match = re.search(pattern, line)
            if match:
                # If we found a new timestamp, save previous and start new
                if current_clip:
                    clips.append(current_clip)
                
                start_raw, end_raw, desc_raw = match.groups()
                
                current_clip = {
                    "start": self._parse_timestamp(start_raw),
                    "end": self._parse_timestamp(end_raw),
                    "title": desc_raw.split('.')[0].strip() if desc_raw else "Viral Highlight",
                    "description": desc_raw.strip(),
                    "explanation": desc_raw.strip(),
                    "hashtags": "#viral #highlight"
                }
            elif current_clip and line.strip().startswith(('-', '   ')):
                # Append subsequent bullet points to the description
                clean_line = line.strip().lstrip('-').strip()
                if clean_line:
                    current_clip["description"] += " " + clean_line
                    current_clip["explanation"] += " " + clean_line
        
        if current_clip:
            clips.append(current_clip)
            
        return clips

    def _parse_timestamp(self, ts) -> float:
        """Helper to parse either float or HH:MM:SS string to seconds."""
        if ts is None: return 0.0
        if isinstance(ts, (int, float)): return float(ts)
        
        ts_str = str(ts).strip()
        if not ts_str: return 0.0
        
        # Handle HH:MM:SS or MM:SS
        if ':' in ts_str:
            parts = ts_str.split(':')
            try:
                if len(parts) == 3: # HH:MM:SS
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                elif len(parts) == 2: # MM:SS
                    return int(parts[0]) * 60 + float(parts[1])
            except:
                pass
        
        try:
            return float(ts_str)
        except:
            return 0.0

    def _api_analyze(self, transcript_segments: list[dict]) -> list[dict]:
        """Call OpenRouter API to analyze transcript."""
        if not prefs.openrouter_key:
            raise Exception("OpenRouter API Key is missing. Please set it in Settings.")

        full_transcript = [
            f"[{s['start']:.2f}-{s['end']:.2f}] {self._sanitize_transcript(s['text'])}"
            for s in transcript_segments
        ]
        context_text = "\n".join(full_transcript)

        prompt = (
            f"Analyze this video transcript and identify the most significant 'viral moments'.\n"
            "A viral moment is a segment with a strong hook, a clear emotional peak, or high-value advice.\n"
            "- CRITICAL: Use natural clip lengths (30-90s recommended). Do NOT default to exactly 10 seconds.\n"
            "- EACH clip MUST NOT exceed 180 seconds.\n"
            "- Start early enough for context and end naturally.\n"
            'Return ONLY JSON in this format: [{ "start": 12.5, "end": 48.0, "title": "Hook Name", "description": "...", "hashtags": "#tag1", "score": 9.5 }]\n\n'
            f"Transcript:\n{context_text}"
        )

        headers = {
            "Authorization": f"Bearer {prefs.openrouter_key}",
            "HTTP-Referer": "https://github.com/clipperr", # Required by OpenRouter
            "X-Title": "ClipperR Video Assistant",
            "Content-Type": "application/json"
        }

        payload = {
            "model": prefs.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": prefs.llm_temperature,
            "max_tokens": prefs.llm_max_tokens,
        }

        try:
            log.info("Calling OpenRouter API (%s)...", prefs.openrouter_model)
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            content = data["choices"][0]["message"]["content"].strip()
            
            content = self._repair_json(content)
            
            try:
                clips = json.loads(content)
            except json.JSONDecodeError as jde:
                log.error("Failed to parse API JSON. Repaired content: %s", content)
                raise Exception(f"API produced invalid JSON output. Error: {jde}")
            
            # Normalize
            normalized = []
            for c in clips:
                norm = {
                    "start": c.get("start", 0.0),
                    "end": c.get("end", 0.0),
                    "title": c.get("title", "Viral Moment"),
                    "description": c.get("description", ""),
                    "hashtags": c.get("hashtags", "#viral"),
                    "score": c.get("score", 0.0)
                }
                normalized.append(norm)

            log.info("API analysis found %d clips", len(normalized))
            return normalized

        except Exception as e:
            log.error("OpenRouter API error: %s", e)
            raise Exception(f"OpenRouter API failed: {e}")

    def generate_clickbait(self, script_text: str) -> str:
        """Single-pass clickbait generation."""
        titles = self.batch_generate_clickbait([script_text])
        return titles[0] if titles else ""

    def batch_generate_clickbait(self, scripts: list[str]) -> list[str]:
        """Generates multiple viral titles in ONE model load/unload cycle for memory safety."""
        if not scripts: return []
        
        try:
            prefs.load()
            if prefs.llm_provider == "api":
                return [self._api_clickbait(s) for s in scripts]
            
            # Local Mode: Load once
            if not self.available:
                self.load_model()
            
            if not (self.llm and self.available):
                return ["" for _ in scripts]

            results = []
            for text in scripts:
                prompt = (
                    "You are a viral YouTube Shorts and TikTok thumbnail expert.\n"
                    "Create a hyper-clickbait, extreme, and extremely short title (3-5 words) for the FOLLOWING TRANSCRIPT.\n"
                    "The title MUST be in the same language as the transcript.\n"
                    "Respond ONLY with the text, no quotes or filler.\n\n"
                    f"Transcript: {text[:500]}...\n" # Limit input per clip to save context
                )
                messages = [{"role": "user", "content": prompt}]
                outputs = self.llm(
                    messages,
                    max_new_tokens=30,
                    temperature=0.8,
                    do_sample=True,
                    max_length=None,
                    pad_token_id=self.llm.tokenizer.eos_token_id if hasattr(self.llm, 'tokenizer') else None,
                )
                
                raw_res = outputs[0]["generated_text"]
                if isinstance(raw_res, list) and len(raw_res) > 0:
                    res = raw_res[-1].get("content", "")
                else:
                    res = str(raw_res)
                
                results.append(res.strip().strip('"*\'').upper())
            return results
        except Exception as e:
            log.error("Batch clickbait failed: %s", e)
            return ["" for _ in scripts]
        finally:
            if not prefs.llm_provider == "api":
                self.unload_model()

    def _api_clickbait(self, text: str) -> str:
        """Call API for single clickbait title."""
        prompt = f"Create a viral 3-5 word YouTube Shorts title for this: {text[:500]}"
        headers = {
            "Authorization": f"Bearer {prefs.openrouter_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": prefs.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 30,
            "temperature": 0.8
        }
        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip().strip('"*\'').upper()
        except Exception:
            return ""

    # ══════════════════════════════════════════════════
    #  Heuristic fallback
    # ══════════════════════════════════════════════════
    def _heuristic_analyze(self, transcript_segments: list[dict]) -> list[dict]:
        """Pick clips based on word-density (words per second) in sliding windows."""
        if not transcript_segments:
            return []

        total_duration = transcript_segments[-1]["end"]
        clip_duration = min(60.0, max(30.0, total_duration / 10))

        windows: list[dict] = []
        window_start = 0.0

        while window_start < total_duration:
            window_end = min(window_start + clip_duration, total_duration)

            word_count = 0
            texts: list[str] = []
            for s in transcript_segments:
                if s["start"] >= window_start and s["end"] <= window_end:
                    word_count += len(s["text"].split())
                    texts.append(s["text"])

            duration = window_end - window_start
            density = word_count / max(duration, 1.0)

            windows.append({
                "start": round(window_start, 2),
                "end": round(window_end, 2),
                "density": density,
                "preview": " ".join(texts)[:80],
            })
            window_start += clip_duration / 2  # 50% overlap

        windows.sort(key=lambda x: x["density"], reverse=True)

        clips = []
        for w in windows[:prefs.max_viral_clips]:
            # PRO-EDITOR: More descriptive heuristic titles
            start_time = int(w['start'])
            clips.append({
                "start": w['start'],
                "end": w['end'],
                "title": f"Viral Moment @ {start_time}s",
                "description": f"High engagement highlight from {w['preview'][:40]}...",
                "hashtags": "#clipper #viral #highlights",
                "explanation": f"High engagement segment based on word density.",
                "speaker": "UNKNOWN",
            })

        log.info("Heuristic analysis found %d clips", len(clips))
        return clips
