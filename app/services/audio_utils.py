import subprocess
import math
import struct
from logger import get_logger

log = get_logger(__name__)

# ── Legacy standalone function (backward compat) ──────
def get_audio_energy(video_path: str, timestamp_sec: float, duration_sec: float = 0.2) -> float:
    """
    Extract a small PCM chunk from video at timestamp and return RMS energy (0.0 - 1.0).
    NOTE: For batch operations, use AudioEnergyCache instead — it's 10-50x faster.
    """
    try:
        start_time = max(0, timestamp_sec - 0.05)
        
        cmd = [
            "ffmpeg",
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
            
        count = len(raw_data) // 2
        if count == 0:
            return 0.0
            
        samples = struct.unpack(f"{count}h", raw_data)
        
        sum_sq = sum(float(s)**2 for s in samples)
        rms = math.sqrt(sum_sq / count)
        
        normalized = min(rms / 12000.0, 1.0)
        return normalized
        
    except Exception as e:
        log.warning("Audio energy extraction failed: %s", e)
        return 0.0


# ══════════════════════════════════════════════════════
#  Batch Audio Energy Cache
# ══════════════════════════════════════════════════════
class AudioEnergyCache:
    """
    Cache audio waveform for a time range to avoid repeated FFmpeg subprocess calls.
    
    Usage:
        cache = AudioEnergyCache()
        cache.preload(video_path, start=10.0, end=40.0)
        energy = cache.get_energy(video_path, timestamp=15.5)
        cache.clear()
    """
    
    SAMPLE_RATE = 16000  # 16kHz mono
    RMS_LOUD_THRESHOLD = 12000.0  # Normalization ceiling for speech
    
    def __init__(self):
        self._samples = None       # Raw PCM samples as list of ints
        self._sample_rate = self.SAMPLE_RATE
        self._start_time = 0.0     # The start time of the cached range
        self._end_time = 0.0
        self._cache_key = None     # (path, start, end) tuple for cache validation
        self._cache_video_path = None  # Track which video is cached
    
    def preload(self, video_path: str, start: float, end: float, margin: float = 0.5) -> bool:
        """
        Extract audio for the entire [start-margin, end+margin] range in ONE subprocess call.
        Returns True if successful.
        """
        actual_start = max(0, start - margin)
        actual_duration = (end + margin) - actual_start
        
        cache_key = (video_path, actual_start, actual_duration)
        if self._cache_key == cache_key and self._samples is not None:
            return True  # Already cached
        
        try:
            cmd = [
                "ffmpeg",
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
                log.warning("AudioEnergyCache: No audio data for range %.1f-%.1f", start, end)
                self._samples = []
                self._cache_key = cache_key
                self._start_time = actual_start
                self._end_time = actual_start + actual_duration
                return False
            
            count = len(raw_data) // 2
            self._samples = list(struct.unpack(f"{count}h", raw_data))
            self._cache_key = cache_key
            self._cache_video_path = video_path
            self._start_time = actual_start
            self._end_time = actual_start + actual_duration
            
            log.debug("AudioEnergyCache: Loaded %d samples (%.1fs-%.1fs)", 
                       count, actual_start, self._end_time)
            return True
            
        except Exception as e:
            log.warning("AudioEnergyCache preload failed: %s", e)
            self._samples = []
            self._cache_key = cache_key
            return False
    
    def get_energy(self, timestamp: float, duration: float = 0.2, video_path: str = None) -> float:
        """
        Get RMS energy at timestamp from cached waveform.
        Zero subprocess overhead — just numpy-like list slicing.
        If video_path is provided, validates it matches the cached video.
        """
        if not self._samples:
            return 0.0
        
        # Validate cache matches requested video
        if video_path and self._cache_video_path and video_path != self._cache_video_path:
            log.warning("AudioEnergyCache: video mismatch (cached=%s, requested=%s)",
                        self._cache_video_path, video_path)
            return 0.0
        
        try:
            # Convert timestamp to sample index relative to cache start
            relative_time = timestamp - self._start_time
            if relative_time < 0:
                relative_time = 0.0
            
            start_idx = int(relative_time * self._sample_rate)
            end_idx = int((relative_time + duration) * self._sample_rate)
            
            # Clamp to valid range
            start_idx = max(0, min(start_idx, len(self._samples) - 1))
            end_idx = max(start_idx + 1, min(end_idx, len(self._samples)))
            
            chunk = self._samples[start_idx:end_idx]
            if not chunk:
                return 0.0
            
            # RMS calculation
            sum_sq = sum(float(s) ** 2 for s in chunk)
            rms = math.sqrt(sum_sq / len(chunk))
            
            return min(rms / self.RMS_LOUD_THRESHOLD, 1.0)
            
        except Exception as e:
            log.warning("AudioEnergyCache.get_energy error: %s", e)
            return 0.0
    
    def clear(self):
        """Release cached audio data."""
        self._samples = None
        self._cache_key = None
        self._cache_video_path = None
        self._start_time = 0.0
        self._end_time = 0.0
