import unittest
import os
import sys
import json

# Ensure 'app' directory is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from services.history_manager import HistoryManager

class TestServices(unittest.TestCase):
    def setUp(self):
        self.test_history_file = "test_history.json"
        if os.path.exists(self.test_history_file):
            os.remove(self.test_history_file)
        
        # Override the history file for testing
        self.manager = HistoryManager()
        self.manager.history_file = self.test_history_file

    def test_save_and_load_history(self):
        """Verify that video processing history can be saved and retrieved."""
        # HistoryManager.clips is a list
        test_clip = {
            "path": "test.mp4",
            "output_path": "exports/test_out.mp4",
            "title": "Test Title"
        }
        
        # Add and Save
        self.manager.clips = [test_clip]
        self.manager.save()
        self.assertTrue(os.path.exists(self.test_history_file))
        
        # Create new manager to check load
        new_manager = HistoryManager()
        new_manager.history_file = self.test_history_file
        loaded_clips = new_manager.load()
        self.assertEqual(len(loaded_clips), 1)
        self.assertEqual(loaded_clips[0]["path"], "test.mp4")

    def tearDown(self):
        if os.path.exists(self.test_history_file):
            os.remove(self.test_history_file)

if __name__ == '__main__':
    unittest.main()
