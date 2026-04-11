import sys
import os

# Get project root (two levels up from this file)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'app'))

from services.video_processor import VideoProcessor

def mock_callback(msg, progress):
    print(f"[PROGRESS] {progress}%: {msg}")

def test_callback_reference():
    vp = VideoProcessor()
    print("Testing _get_dynamic_focus callback reference...")
    try:
        # We don't actually need to run it on a real video to check if the variable is defined
        # But we can check if the method signature is correct
        import inspect
        sig = inspect.signature(vp._get_dynamic_focus)
        if 'progress_callback' in sig.parameters:
            print("✅ progress_callback is in _get_dynamic_focus signature")
        else:
            print("❌ progress_callback MISSING from _get_dynamic_focus signature")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_callback_reference()
