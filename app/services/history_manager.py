import json
import os
from config import HISTORY_FILE
from logger import get_logger

log = get_logger(__name__)

class HistoryManager:
    """Manages the persistent list of clipped videos."""

    def __init__(self):
        self.history_file = HISTORY_FILE
        self.clips = self.load()

    def load(self) -> list[dict]:
        """Load clips from history.json."""
        if not os.path.exists(self.history_file):
            return []
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            log.error("Failed to load history: %s", e)
            return []

    def save(self):
        """Save the current clips list to history.json."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.clips, f, indent=4)
        except Exception as e:
            log.error("Failed to save history: %s", e)

    MAX_HISTORY = 100  # Prevent unbounded growth

    def add_clips(self, new_clips: list[dict]):
        """Add new clips to the top of the history and save."""
        # Prepend so new items show first
        self.clips = new_clips + self.clips
        # Cap history size — drop oldest entries
        if len(self.clips) > self.MAX_HISTORY:
            self.clips = self.clips[:self.MAX_HISTORY]
        self.save()

    def remove_clip(self, output_path: str):
        """Remove a clip from history by its output path and save."""
        original_count = len(self.clips)
        self.clips = [c for c in self.clips if c.get('output_path') != output_path]
        if len(self.clips) < original_count:
            self.save()
            log.info("Removed clip from history: %s", output_path)
        else:
            log.warning("Clip not found in history: %s", output_path)

    def clear(self):
        """Clear all history."""
        self.clips = []
        self.save()
