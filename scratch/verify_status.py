import sys
import os

# Add app to path
sys.path.append(os.path.join(os.getcwd(), 'app'))

from services.downloader import ModelManager

mm = ModelManager()

print(f"Checking for whisper-base at: {mm.models['whisper-base']['path']}")
exists = mm.check_status("whisper-base")
print(f"Whisper exists: {exists}")

print(f"Checking for llm-analysis at: {mm.models['llm-analysis']['path']}")
llm_exists = mm.check_status("llm-analysis")
print(f"LLM exists: {llm_exists}")
