import subprocess
import math
import struct
import numpy as np
from logger import get_logger

log = get_logger(__name__)

# ── Legacy standalone function (backward compat) ──────
def get_audio_energy(video_path: str, timestamp_sec: float, duration_sec: float = 0.2) -> float:
    """
    Extract a small PCM chunk from video at timestamp and return RMS energy (0.0 - 1.0).
    NOTE: For batch operations, use AudioEnergyCache instead — it's 10-50x faster.
    """
    try:
        import config
        start_time = max(0, timestamp_sec - 0.05)
        
        cmd = [
            config.FFMPEG_EXE,
            "-ss", str(start_time),
            "-t", str(duration_sec),
            "-i", video_path,
            "-f", "s16le",
            "-ac", "1",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-loglevel", "quiet",
            "-"
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_data, _ = process.communicate(timeout=2.0)
        
        if not raw_data:
            return 0.0
            
        # Optimization: Use numpy even for single chunk
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            return 0.0
            
        rms = np.sqrt(np.mean(np.square(samples)))
        return min(rms / 12000.0, 1.0)
        
    except Exception as e:
        log.warning("Audio energy extraction failed: %s", e)
        return 0.0


# ══════════════════════════════════════════════════════
#  Batch Audio Energy Cache (NumPy Optimized)
# ══════════════════════════════════════════════════════
class AudioEnergyCache:
    """
    Cache audio waveform for a time range to avoid repeated FFmpeg subprocess calls.
    Uses NumPy for memory efficiency (10x reduction) and vectorized operations.
    """
    
    SAMPLE_RATE = 16000  # 16kHz mono
    RMS_LOUD_THRESHOLD = 12000.0  # Normalization ceiling for speech
    
    def __init__(self):
        self._samples = None       # Raw pcm samples as np.ndarray (int16)
        self._sample_rate = self.SAMPLE_RATE
        self._start_time = 0.0
        self._end_time = 0.0
        self._cache_key = None
        self._cache_video_path = None
    
    def preload(self, video_path: str, start: float, end: float, margin: float = 0.5) -> bool:
        """Extract audio for the entire range in ONE call."""
        actual_start = max(0, start - margin)
        actual_duration = (end + margin) - actual_start
        
        cache_key = (video_path, actual_start, actual_duration)
        if self._cache_key == cache_key and self._samples is not None:
            return True
            
        try:
            from config import FFMPEG_EXE
            cmd = [
                FFMPEG_EXE,
                "-ss", str(actual_start),
                "-t", str(actual_duration),
                "-i", video_path,
                "-f", "s16le",
                "-ac", "1",
                "-ar", str(self._sample_rate),
                "-acodec", "pcm_s16le",
                "-loglevel", "quiet",
                "-"
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            raw_data, _ = process.communicate(timeout=max(10.0, actual_duration * 2))
            
            if not raw_data:
                self._samples = np.array([], dtype=np.int16)
                self._cache_key = cache_key
                return False
            
            # AU-3 Fix: Use np.frombuffer for zero-copy-like efficiency
            self._samples = np.frombuffer(raw_data, dtype=np.int16).copy()
            self._cache_key = cache_key
            self._cache_video_path = video_path
            self._start_time = actual_start
            self._end_time = actual_start + actual_duration
            
            log.debug("AudioEnergyCache: Loaded %d samples (memory optimized)", len(self._samples))
            return True
            
        except Exception as e:
            log.warning("AudioEnergyCache preload failed: %s", e)
            self._samples = np.array([], dtype=np.int16)
            self._cache_key = cache_key
            return False
    
    def get_energy(self, timestamp: float, duration: float = 0.2, video_path: str = None) -> float:
        """Get RMS energy using vectorized operations."""
        if self._samples is None or len(self._samples) == 0:
            return 0.0
        
        if video_path and self._cache_video_path and video_path != self._cache_video_path:
            return 0.0
        
        try:
            relative_time = timestamp - self._start_time
            if relative_time < 0: relative_time = 0.0
            
            start_idx = int(relative_time * self._sample_rate)
            end_idx = int((relative_time + duration) * self._sample_rate)
            
            # Slice and calculate RMS via NumPy
            chunk = self._samples[start_idx:end_idx].astype(np.float32)
            if len(chunk) == 0:
                return 0.0
            
            rms = np.sqrt(np.mean(np.square(chunk)))
            return min(rms / self.RMS_LOUD_THRESHOLD, 1.0)
            
        except Exception as e:
            log.warning("AudioEnergyCache.get_energy error: %s", e)
            return 0.0
    
    def clear(self):
        self._samples = None
        self._cache_key = None
        self._cache_video_path = None
