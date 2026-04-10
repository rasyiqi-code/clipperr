import unittest
import os
import json
import sys

# Ensure 'app' directory is in the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from config import UserSettings, USER_SETTINGS_FILE

class TestConfig(unittest.TestCase):
    def setUp(self):
        # Use a temporary settings file path for testing
        self.test_file = "test_user_settings.json"
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
            
        # We manually inject the test file path for this test run
        import config
        self.original_path = config.USER_SETTINGS_FILE
        config.USER_SETTINGS_FILE = self.test_file

    def test_settings_default_values(self):
        """Verify that a new UserSettings instance has correct defaults."""
        settings = UserSettings()
        # Now it should be top_left because it's a fresh file/instance
        self.assertEqual(settings.watermark_pos, "top_left")
        self.assertEqual(settings.llm_provider, "local")

    def test_settings_persistence_logic(self):
        """Verify that data can be saved and loaded accurately."""
        settings = UserSettings()
        settings.openrouter_key = "sk-test-123"
        settings.save()
        
        # Load into a new instance
        new_settings = UserSettings()
        self.assertEqual(new_settings.openrouter_key, "sk-test-123")

    def tearDown(self):
        import config
        config.USER_SETTINGS_FILE = self.original_path
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

if __name__ == '__main__':
    unittest.main()
