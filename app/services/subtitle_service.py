import os
import re
from datetime import timedelta

from logger import get_logger

log = get_logger(__name__)

class SubtitleService:
    """
    Generates Advanced SubStation Alpha (.ass) files from transcription segments.
    Provides TikTok/Shorts style dynamic word-level highlighting and chunking.
    """

    def __init__(self):
        # Speaker colors in ASS format: &HBBGGRR& (Hex blue, green, red without alpha for the \c tag)
        self.speaker_colors = {
            "UNKNOWN": "&H00D7FF&",    # Yellow
            "SPEAKER_00": "&H00D7FF&", # Yellow
            "SPEAKER_01": "&HFFFF00&", # Cyan
            "SPEAKER_02": "&H00FF00&", # Green
            "SPEAKER_03": "&HFF00FF&", # Magenta
        }

    def _format_time_ass(self, seconds: float) -> str:
        """Converts seconds to ASS format: H:MM:SS.cs"""
        td = timedelta(seconds=max(0.0, float(seconds)))
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        centis = int((td.microseconds / 1000000) * 100)
        return f"{hours}:{minutes:02}:{secs:02}.{centis:02}"

    def _get_ass_header(self) -> str:
        """Returns the ASS style header with robust TikTok-style formatting."""
        return """[Script Info]
ScriptType: v4.00+
PlayResX: 608
PlayResY: 1080
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Sans,45,&H00FFFFFF,&H000000FF,&H00000000,&H99000000,-1,0,0,0,100,100,0,0,1,3,4,2,30,30,160,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def _chunk_words(self, words: list[dict], max_words=3, max_chars=20) -> list[list[dict]]:
        """Groups words into short chunks for punchy short-form video subtitles."""
        chunks = []
        current_chunk = []
        current_len = 0
        
        for w in words:
            text = w.get("word", "").strip()
            if not text:
                continue
            
            # Start new chunk if limit reached
            if current_chunk and (len(current_chunk) >= max_words or current_len + len(text) > max_chars):
                chunks.append(current_chunk)
                current_chunk = []
                current_len = 0
            
            # Check for sentence end markers (force break after this word)
            force_break = bool(re.search(r'[.!?]$', text))
            
            current_chunk.append(w)
            current_len += len(text) + 1 # +1 for space
            
            if force_break:
                chunks.append(current_chunk)
                current_chunk = []
                current_len = 0
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def generate_ass(self, segments: list[dict], clip_start: float, clip_end: float, output_path: str):
        """
        Creates a dynamic ASS file with word-level highlighting.
        """
        log.info("Generating ASS for clip [%.2f - %.2f] -> %s", clip_start, clip_end, output_path)
        
        if not output_path.endswith('.ass'):
            output_path = output_path.rsplit('.', 1)[0] + '.ass'
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self._get_ass_header())
            
            line_count = 0
            
            for seg in segments:
                # Handle raw dicts or objects
                start = seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", None)
                end = seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", None)
                speaker = seg.get("speaker", "UNKNOWN") if isinstance(seg, dict) else getattr(seg, "speaker", "UNKNOWN")
                words = seg.get("words", []) if isinstance(seg, dict) else getattr(seg, "words", [])

                if start is None or end is None:
                    continue

                if start >= clip_end or end <= clip_start:
                    continue

                # Get highlighting color based on speaker
                hl_color = self.speaker_colors.get(speaker, self.speaker_colors["UNKNOWN"])

                if not words:
                    # Fallback to normal rendering if no word-level timestamps exist
                    rel_start = max(0.0, start - clip_start)
                    rel_end = min(clip_end - clip_start, end - clip_start)
                    text = seg.get("text", "").strip() if isinstance(seg, dict) else getattr(seg, "text", "").strip()
                    if text and rel_start < rel_end:
                        f.write(f"Dialogue: 0,{self._format_time_ass(rel_start)},{self._format_time_ass(rel_end)},Default,,0,0,0,,{text}\n")
                        line_count += 1
                    continue

                # Dynamic Chunking
                chunks = self._chunk_words(words)
                
                for chunk in chunks:
                    chunk_start = chunk[0]["start"]
                    chunk_end = chunk[-1]["end"]
                    
                    if chunk_start >= clip_end or chunk_end <= clip_start:
                        continue
                        
                    # Calculate timeline bounds for chunk relative to clip
                    rel_c_start = max(0.0, chunk_start - clip_start)
                    rel_c_end = min(clip_end - clip_start, chunk_end - clip_start)
                    
                    if rel_c_start >= rel_c_end:
                        continue

                    # In ASS, we don't duplicate line events per word. We use karaoke tags or 
                    # create overlapping events. The simplest aesthetic way is overlapping events:
                    # For each word in the chunk, we create an event showing the FULL chunk, 
                    # but only that specific word is colored.
                    # This achieves the "karaoke pop" effect cleanly.
                    
                    for i, active_word in enumerate(chunk):
                        w_start = active_word["start"]
                        w_end = active_word["end"]
                        
                        rel_w_start = max(0.0, w_start - clip_start)
                        rel_w_end = min(clip_end - clip_start, w_end - clip_start)
                        
                        # Ensure duration is valid
                        if rel_w_start >= rel_w_end:
                            continue
                            
                        # Build the line text with highlight tags
                        styled_text = ""
                        for j, word_blob in enumerate(chunk):
                            w_text = word_blob["word"].strip()
                            if not w_text: continue
                            
                            if j == i:
                                # Highlight this word
                                styled_text += f"{{\\c{hl_color}}}{w_text}{{\\c&HFFFFFF&}} "
                            else:
                                styled_text += f"{w_text} "
                                
                        styled_text = styled_text.strip()
                        
                        event_line = f"Dialogue: 0,{self._format_time_ass(rel_w_start)},{self._format_time_ass(rel_w_end)},Default,,0,0,0,,{styled_text}\n"
                        f.write(event_line)
                        line_count += 1
                        
        log.info("ASS generated with %d dynamic subtitle events.", line_count)
        return output_path

