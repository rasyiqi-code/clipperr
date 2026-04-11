import unittest
import shutil
import os
import sys

# Ensure 'app' directory is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

class TestSystem(unittest.TestCase):
    def test_ffmpeg_available(self):
        """Verify that ffmpeg is installed and available in the system PATH."""
        ffmpeg_path = shutil.which("ffmpeg")
        self.assertIsNotNone(ffmpeg_path, "FFmpeg is NOT installed or not in PATH. AI rendering will fail.")
        print(f"  [System] FFmpeg found at: {ffmpeg_path}")

    def test_ffprobe_available(self):
        """Verify that ffprobe is installed and available."""
        ffprobe_path = shutil.which("ffprobe")
        self.assertIsNotNone(ffprobe_path, "FFprobe is NOT installed. Media metadata analysis will fail.")

    def test_core_engine_import(self):
        """Verify that the Rust core component (clipperr_core) is correctly installed."""
        try:
            import clipperr_core
            self.assertTrue(hasattr(clipperr_core, '__file__'), "clipperr_core imported but has no file attribute.")
            print(f"  [System] clipperr_core found at: {clipperr_core.__file__}")
        except ImportError:
            self.fail("clipperr_core is NOT installed. Run 'maturin build' in core-engine/ to fix this.")

if __name__ == '__main__':
    unittest.main()
