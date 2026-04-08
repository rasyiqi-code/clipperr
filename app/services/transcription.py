import os

from config import TRANSCRIPTION_BEAM_SIZE
from logger import get_logger

log = get_logger(__name__)


class TranscriptionService:
    def __init__(self, model_path: str = "models/whisper/base", device=None):
        self.device = device
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """
        Load the model only if files are present locally.
        Lazy-imports torch and faster_whisper so the UI can start without them.
        """
        import torch
        from faster_whisper import WhisperModel

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.compute_type = "int8" if self.device == "cpu" else "float16"

        model_bin = os.path.join(self.model_path, "model.bin")
        if not os.path.exists(model_bin):
            raise FileNotFoundError(
                f"Whisper model not found at {self.model_path}. "
                "Please download it in Settings."
            )

        log.info("Loading Whisper model from %s on %s (%s)", self.model_path, self.device, self.compute_type)
        self.model = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
            local_files_only=True,
        )

    def transcribe(self, audio_path: str, language: str = "id", progress_callback=None):
        """Transcribe *audio_path* and return (segments_list, info)."""
        if not self.model:
            self.load_model()

        # 1. Transcribe (Whisper)
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=TRANSCRIPTION_BEAM_SIZE,
            language=language,
            word_timestamps=True,
        )

        duration = info.duration if info.duration else 1.0
        results = []
        for segment in segments:
            seg_dict = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "speaker": "UNKNOWN",
                "words": []
            }
            
            if hasattr(segment, 'words') and segment.words:
                for w in segment.words:
                    seg_dict["words"].append({
                        "start": w.start,
                        "end": w.end,
                        "word": w.word,
                        "prob": w.probability
                    })
            
            results.append(seg_dict)
            
            if progress_callback:
                pct = min(int((segment.end / duration) * 100), 100)
                progress_callback(f"Transcribing... {pct}%", 15 + int(pct * 0.35))

        log.info("Transcription complete: %d segments", len(results))
        return results, info
