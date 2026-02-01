import pandas as pd
import Levenshtein
import re
import json
from collections import Counter

# --- Configuration ---
FILE_PATH = '../poems_translated.parquet'
VOCAB_COLUMN = 'text'

# Azerbaijani Text Cleaning ---
def az_lowercase(text):
    """
    Handles specific Azerbaijani casing issues (I -> ı, İ -> i).
    Standard .lower() maps I -> i which is incorrect for Azerbaijani.
    """
    if not isinstance(text, str):
        return ""
    
    # Map capital I to small dotless ı, and capital İ to small i
    text = text.replace('I', 'ı').replace('İ', 'i')
    return text.lower()

def preprocess_text(text):
    """
    Cleans text: lowercases, removes punctuation/numbers, splits into tokens.
    """
    text = az_lowercase(text)
    # Remove punctuation and numbers, keep only Azerbaijani alphabet chars
    text = re.sub(r'[^\w\s]', '', text) 
    return text.split()

# --- Step 1: Load and Build Vocabulary ---
print("Loading Parquet file...")
try:
    df = pd.read_parquet(FILE_PATH)
    print(f"Loaded {len(df)} rows.")
except FileNotFoundError:
    # Creating dummy data for demonstration if you run this without the file
    print("File not found")
    exit(0)

# Build the Frequency Dictionary
print("Building Vocabulary...")
all_words = []

# iterate over the modern_text column to build the valid word bank
for line in df[VOCAB_COLUMN]:
    tokens = preprocess_text(line)
    all_words.extend(tokens)

# Create a vocabulary set (unique words) and a frequency counter
vocab_counts = Counter(all_words)
vocab_set = set(vocab_counts.keys())

print(f"Vocabulary built. Total unique words: {len(vocab_set)}")

# The Levenshtein Checker System

def get_closest_word(word, vocab_set, max_distance=3):
    """
    Finds the closest word in the vocabulary using Levenshtein distance.
    Returns: (best_match, distance)
    """
    word = az_lowercase(word)
    
    # 1. Exact match check
    if word in vocab_set:
        return word, 0
    
    # 2. Levenshtein Search
    best_word = None
    min_dist = float('inf')
    
    for candidate in vocab_set:
        # Skip words with large length differences
        if abs(len(candidate) - len(word)) > max_distance:
            continue
            
        dist = Levenshtein.distance(word, candidate)
        
        if dist < min_dist:
            min_dist = dist
            best_word = candidate
            
        # Early exit for very good matches
        if min_dist == 1: 
            break
            
    # Return best match if it is within the acceptable threshold
    if min_dist <= max_distance:
        return best_word, min_dist
    else:
        return None, min_dist

# Testing the System

print("\nTesting the Spelling Checker")

input_file = 'spelling_tests.json'
output_file = 'test_results.json'

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find {input_file}. Please run the generator script first.")
    test_data = []

results = []
correct_count = 0

# 2. Run the checker on each test case
for case in test_data:
    input_word = case['input']
    expected_word = case['expected']
    error_type = case.get('type', 'unknown')
    
    # Run the Levenshtein logic
    # Note: Ensure get_closest_word and vocab_set are defined from Step 1 & 2
    suggestion, dist = get_closest_word(input_word, vocab_set)
    
    # Determine success
    is_correct = (suggestion == expected_word)
    if is_correct:
        correct_count += 1
        
    # Create result entry
    result_entry = {
        "input_word": input_word,
        "expected_word": expected_word,
        "predicted_word": suggestion,  # Will be None (null in JSON) if no match found
        "levenshtein_distance": dist,
        "status": "PASS" if is_correct else "FAIL",
        "error_type": error_type
    }
    
    results.append(result_entry)

# 3. Save the results to a new JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# 4. Print a brief summary to console
if test_data:
    accuracy = (correct_count / len(test_data)) * 100
    print(f"Processing complete.")
    print(f"Total Tests: {len(test_data)}")
    print(f"Passed:      {correct_count}")
    print(f"Failed:      {len(test_data) - correct_count}")
    print(f"Accuracy:    {accuracy:.2f}%")
    print(f"Detailed results saved to '{output_file}'")
else:
    print("No test data found.")