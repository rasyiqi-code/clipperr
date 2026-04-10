import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

from config import prefs
from logger import get_logger
from services.analysis import AnalysisService

log = get_logger(__name__)


class ThumbnailService:
    def __init__(self, analysis_service: AnalysisService = None):
        self.analysis_service = analysis_service

    def generate_thumbnail(self, video_path: str, extract_time: float, clip_text: str, output_path: str, center_x_norm: float = 0.5, pre_generated_title: str = None):
        """
        Extract a frame from the original video, ask LLM for clickbait text, and overlay it.
        """
        if not prefs.auto_thumbnail:
            return None

        log.info("Generating AI thumbnail for %s", os.path.basename(video_path))
        
        # Ensure outputs directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        raw_frame_path = output_path.replace(".jpg", "_raw.jpg")

        # 1. Extract frame
        if not self._extract_frame(video_path, extract_time, center_x_norm, raw_frame_path):
            return None

        # 2. Get clickbait text
        clickbait_text = self._generate_clickbait(clip_text, pre_generated_title)
        
        # 3. Draw text and save
        success = self._draw_thumbnail_text(raw_frame_path, output_path, clickbait_text)
        
        # Cleanup raw frame
        if os.path.exists(raw_frame_path):
            try:
                os.remove(raw_frame_path)
            except Exception:
                pass
                
        return output_path if success else None

    def _extract_frame(self, video_path: str, extract_time: float, center_x_norm: float, out_path: str) -> bool:
        """Extract a single cropped 9:16 frame at the given timestamp."""
        try:
            # Reconstruct the ffmpeg crop logic used in lib.rs
            # crop=1080:1920:iw*0.5-ih*9/16/2:0
            # Clamp center_x_norm to safe range (matches lib.rs build_static_crop)
            cx = max(0.05, min(0.95, center_x_norm))
            # RS-2 fix: Boundary clamp with max(0, min(iw-crop_w, expr))
            crop_filter = f"crop=ih*9/16:ih:max(0\\,min(iw-ih*9/16\\,iw*{cx}-ih*9/16/2)):0,scale=1080:1920"

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(extract_time),
                "-i", video_path,
                "-vframes", "1",
                "-vf", crop_filter,
                "-q:v", "2",  # High quality jpeg
                "-loglevel", "error",
                out_path
            ]
            subprocess.run(cmd, check=True)
            return os.path.exists(out_path)
        except Exception as e:
            log.error("Failed to extract thumbnail frame: %s", e)
            return False

    def _generate_clickbait(self, script_text: str, pre_generated_title: str = None) -> str:
        """Create a hyper-clickbait title from the transcript."""
        # 0. Use pre-generated title if provided (Memory Optimization)
        if pre_generated_title:
            return pre_generated_title.upper()

        # 1. Try to use LLM if available (Fallback)
        if self.analysis_service:
            title = self.analysis_service.generate_clickbait(script_text)
            if title:
                return title
        
        # 2. Smarter Fallback Logic (if LLM fails or is unavailable)
        # Avoid "ngaco" text by selecting better words than just the first 4
        words = [w for w in script_text.split() if len(w) > 3] # Keep meaningful words
        if not words:
            return "WAKTU NYA KLIP!" # Generic fallback
            
        # Try to find a 'hook' sentence (ending in ? or !)
        import re
        sentences = re.split(r'[.!?]', script_text)
        hooks = [s.strip() for s in sentences if '?' in s or '!' in s]
        
        if hooks:
            hook_words = hooks[0].split()
            title = " ".join(hook_words[:4]).upper()
            title = re.sub(r'[.,!?]+$', '', title).replace(',', '')
            return title + "?!"
            
        # If no hook sentence, take 3-4 interesting words from the middle of the clip
        mid = len(words) // 2
        title_words = words[mid:mid+4]
        if not title_words:
            title_words = words[:4]
            
        title = " ".join(title_words).upper()
        title = re.sub(r'[.,!?]+$', '', title).replace(',', '').replace('.', '')
        return title + "!"

    def _draw_thumbnail_text(self, in_path: str, out_path: str, text: str) -> bool:
        try:
            img = Image.open(in_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            # 1. THEME & TYPOGRAPHY (Shorts Style: Bold, Heavy)
            target_w, target_h = 1080, 1920
            if img.size != (target_w, target_h):
                img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                draw = ImageDraw.Draw(img)

            main_font_size = 100 
            sub_font_size = int(main_font_size / 1.618) 
            
            font_paths = [
                "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "DejaVuSans-Bold.ttf"
            ]
            
            font = None
            sub_font = None
            for fp in font_paths:
                try:
                    font = ImageFont.truetype(fp, main_font_size)
                    sub_font = ImageFont.truetype(fp, sub_font_size)
                    break
                except IOError:
                    continue
            
            if not font:
                font = ImageFont.load_default()
                sub_font = font

            # 2. TEXT TRIMMING (Max 5 words)
            raw_words = text.split()[:5]
            if len(raw_words) > 3:
                lines = [(" ".join(raw_words[:2]), font), (" ".join(raw_words[2:]), sub_font)]
            else:
                lines = [(" ".join(raw_words), font)]

            # 3. POSITIONING (Safe Area Zone: Center-Middle)
            total_h = 0
            for line_text, f in lines:
                _, _, _, h = draw.textbbox((0, 0), line_text, font=f)
                total_h += h + 30 
            
            y_offset = (target_h // 2) - (total_h // 2) - 100 
            
            for line_text, f in lines:
                _, _, w, h = draw.textbbox((0, 0), line_text, font=f)
                x_offset = (target_w - w) // 2
                
                # 4. HIGH CONTRAST (Thick Black Outline)
                stroke_width = 12
                for dx in range(-stroke_width, stroke_width+1):
                    for dy in range(-stroke_width, stroke_width+1):
                        if dx*dx + dy*dy <= stroke_width*stroke_width:
                            draw.text((x_offset+dx, y_offset+dy), line_text, font=f, fill="black")
                
                text_color = "yellow" if f == font else "white"
                draw.text((x_offset, y_offset), line_text, font=f, fill=text_color)
                y_offset += h + 30 

            img.save(out_path, quality=95)
            return True
        except Exception as e:
            log.error("Failed to draw thumbnail text: %s", e)
            return False
