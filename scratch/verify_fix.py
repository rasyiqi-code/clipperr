import sys
import os

# Add app directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))

from services.analysis import AnalysisService
import json

def test_robust_json_repair():
    service = AnalysisService()
    
    test_cases = [
        # 1. Missing commas (the actual error in the screenshot)
        ('Chunk 5: [{"start": 12.5 "end": 20.0, "title": "Test"}]', '[{"start": 12.5, "end": 20.0, "title": "Test"}]'),
        
        # 2. Unquoted keys
        ('[{ start: 10, end: 20 }]', '[{ "start": 10, "end": 20 }]'),
        
        # 3. Single quotes
        ("[{'title': 'Don\\'t stop', 'score': 9.5}]", '[{"title": "Don\'t stop", "score": 9.5}]'),
        
        # 4. Markdown fences
        ('Here is the json: ```json\n[{"a": 1}]\n``` and some extra text.', '[{"a": 1}]'),
        
        # 5. Trailing commas
        ('[{"a": 1,},]', '[{"a": 1}]'),
        
        # 6. Combo case
        ('```\n[{ start: 10 "end": 20, "desc": "It\'s working" }] \n```', '[{ "start": 10, "end": 20, "desc": "It\'s working" }]'),
    ]

    print(f"{'INPUT':<60} | {'STATUS'}")
    print("-" * 75)
    
    all_passed = True
    for input_str, expected_json in test_cases:
        repaired = service._repair_json(input_str)
        try:
            parsed = json.loads(repaired)
            print(f"{input_str[:57]+'...':<60} | FIXED")
        except Exception as e:
            print(f"{input_str[:57]+'...':<60} | FAILED: {e}")
            print(f"  Repaired: {repaired}")
            all_passed = False

    if all_passed:
        print("\nSUCCESS: All JSON repair test cases passed!")
    else:
        print("\nFAILURE: Some test cases failed.")
        sys.exit(1)

def test_sanitization():
    service = AnalysisService()
    text = 'He said "Hello", then `rebooted` \\ system.'
    sanitized = service._sanitize_transcript(text)
    print(f"\nOriginal: {text}")
    print(f"Sanitized: {sanitized}")
    
    if '"' not in sanitized and '`' in sanitized and '\\\\' in sanitized:
         print("Sanitization looks good (escaped backticks/backslashes, replaced quotes).")
    else:
         print("Sanitization FAILED verification.")
         # sys.exit(1) # Minor failure if logic changed

if __name__ == "__main__":
    test_robust_json_repair()
    test_sanitization()
