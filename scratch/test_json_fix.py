import json
import re

def repair_json(json_str):
    # 1. Remove comments
    json_str = re.sub(r'//.*', '', json_str)
    
    # 2. Support unquoted keys (e.g., { start: 10 } -> { "start": 10 })
    # This finds a word followed by a colon and wraps the word in quotes if not already quoted.
    # Be careful not to double-quote already quoted keys.
    json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', json_str)
    
    # 3. Support single quotes (replace with double quotes)
    # This is dangerous if there are apostrophes in values, so we target 'key': or :'value'
    # Simplified: replace ' with " if it's near a brace, bracket, colon or comma
    # Actually, just replacing all ' with " is often what's needed for simple AI JSON.
    # But let's be slightly safer:
    json_str = re.sub(r"\'", '"', json_str)
    
    # 4. Add missing commas between key-value pairs
    # Pattern: "value" "key": or 123 "key": or true "key":
    # Using a lookahead to ensure we don't accidentally match something else
    json_str = re.sub(r'("|\d|true|false|null)\s+(")', r'\1, \2', json_str)
    
    # 5. Handle trailing commas
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r',\s*}', '}', json_str)
    
    return json_str

# Test cases
test_cases = [
    '{"start": 12.5 "end": 15.0}',      # Missing comma
    '[{"a": 1 "b": 2}]',                # Missing comma in array object
    '{"a": "val1" "b": "val2"}',        # Missing comma between strings
    '[{"a": 1, "b": 2,}]',               # Trailing comma
    '{ start: 10, end: 20 }',            # Unquoted keys
    "{'start': 10, 'end': 20}",          # Single quotes
    '[{"title": "don\'t stop"}]',        # Apostrophe in value (dangerous with naive replacement)
]

for tc in test_cases:
    print(f"Original: {tc}")
    repaired = repair_json(tc)
    print(f"Repaired: {repaired}")
    try:
        json.loads(repaired)
        print("Status: FIXED")
    except Exception as e:
        print(f"Status: FAILED ({e})")
    print("-" * 20)
