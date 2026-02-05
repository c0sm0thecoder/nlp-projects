import pandas as pd
import numpy as np
import re
import json
import random
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# File paths and settings
FILE_PATH = '../poems_translated.parquet'
TEXT_COLUMN = 'text'
OUTPUT_FILE = 'spell_check_results.json'
CONFUSION_MATRIX_FILE = 'confusion_matrix.json'
CHARACTER_CONFUSION_FILE = 'character_confusion_matrix.json'

def az_lowercase(text):
    # Fix Azerbaijani letters - I becomes ı, İ becomes i
    if not isinstance(text, str):
        return ""
    
    # Map capital I to small dotless ı, and capital İ to small i
    text = text.replace('I', 'ı').replace('İ', 'i')
    return text.lower()

def preprocess_text(text):
    # Clean text and split into words
    text = az_lowercase(text)
    # Remove punctuation and numbers, keep only alphabet chars
    text = re.sub(r'[^\w\s]', '', text) 
    return [word for word in text.split() if len(word) > 0]

# Load the poetry dataset
print("Loading Parquet file...")
try:
    df = pd.read_parquet(FILE_PATH)
    print(f"Loaded {len(df)} rows.")
except FileNotFoundError:
    print("Error: poems_translated.parquet not found!")
    exit(1)

print("Building vocabulary from text...")
all_words = []

# Get all words from the poems
for text in df[TEXT_COLUMN]:
    tokens = preprocess_text(text)
    all_words.extend(tokens)

# Count how often each word appears
vocab_counts = Counter(all_words)
# Only use common words to speed things up
MIN_FREQUENCY = 3
frequent_words = {word for word, count in vocab_counts.items() if count >= MIN_FREQUENCY and len(word) >= 2}
vocab_set = frequent_words

print(f"Total words collected: {len(all_words)}")
print(f"Unique vocabulary size: {len(vocab_counts)}")
print(f"Filtered vocabulary size (freq >= {MIN_FREQUENCY}): {len(vocab_set)}")

# Use all collected words without duplication
print(f"Using {len(all_words)} total words from dataset")

# Edit distance calculator with custom costs for Azerbaijani
class WeightedEditDistance:
    
    def __init__(self):
        # Set up costs for different operations
        self.substitution_costs = self._build_substitution_matrix()
        self.insertion_cost = 1.0
        self.deletion_cost = 1.0
    
    def _build_substitution_matrix(self):
        # Make similar Azerbaijani letters cost less to substitute
        # Groups of similar Azerbaijani letters
        similar_groups = [
            ['a', 'ə'],
            ['o', 'ö'], 
            ['u', 'ü'],
            ['i', 'ı'],
            ['s', 'ş'],
            ['c', 'ç'],
            ['g', 'ğ'],
        ]
        
        costs = defaultdict(lambda: 1.0)
        
        # Make similar letters cheaper to substitute
        for group in similar_groups:
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    char1, char2 = group[i], group[j]
                    costs[(char1, char2)] = 0.5
                    costs[(char2, char1)] = 0.5
        
        return costs
    
    def compute_distance(self, str1, str2):
        # Calculate edit distance between two words
        str1, str2 = az_lowercase(str1), az_lowercase(str2)
        m, n = len(str1), len(str2)
        
        # Create table for dynamic programming
        dp = [[0.0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Fill in starting values
        for i in range(m + 1):
            dp[i][0] = i * self.deletion_cost
        for j in range(n + 1):
            dp[0][j] = j * self.insertion_cost
        
        # Fill the rest of the table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    # Same letter, no cost
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Try different operations and pick cheapest
                    sub_cost = self.substitution_costs.get((str1[i-1], str2[j-1]), 1.0)
                    
                    substitution = dp[i-1][j-1] + sub_cost
                    insertion = dp[i][j-1] + self.insertion_cost
                    deletion = dp[i-1][j] + self.deletion_cost
                    
                    dp[i][j] = min(substitution, insertion, deletion)
        
        return dp[m][n]

# Character-Level Analysis Functions
def get_character_substitutions(error_word, correct_word):
    # Find which letters got changed between two words
    error_word = az_lowercase(error_word)
    correct_word = az_lowercase(correct_word)
    
    # Simple comparison letter by letter
    substitutions = []
    min_len = min(len(error_word), len(correct_word))
    
    # Check each position
    for i in range(min_len):
        if error_word[i] != correct_word[i]:
            substitutions.append((error_word[i], correct_word[i]))
    
    return substitutions

def build_character_confusion_matrix(test_cases):
    # Count how often each letter gets mistaken for another
    char_confusion = defaultdict(int)
    
    for case in test_cases:
        original = case["original"]
        error_word = case["error_word"]
        error_type = case["error_type"]
        
        # Only process substitution errors for character confusion
        if error_type == "substitution":
            substitutions = get_character_substitutions(error_word, original)
            for error_char, correct_char in substitutions:
                # Format: error_char -> correct_char
                char_confusion[(error_char, correct_char)] += 1
    
    return char_confusion

# Error Type Generator
def generate_spelling_errors(word, error_rate=0.3):
    # Create random spelling mistakes for testing
    if len(word) < 2 or random.random() > error_rate:
        return word, "correct"
    
    word = az_lowercase(word)
    error_types = ['substitution', 'insertion', 'deletion', 'transposition']
    error_type = random.choice(error_types)
    
    if error_type == 'substitution' and len(word) > 0:
        pos = random.randint(0, len(word) - 1)
        # Pick a random Azerbaijani letter
        az_chars = 'abcçdeəfgğhxıijklmnoöpqrsştuüvwyz'
        new_char = random.choice(az_chars)
        word = word[:pos] + new_char + word[pos+1:]
        return word, 'substitution'
    
    elif error_type == 'insertion':
        pos = random.randint(0, len(word))
        az_chars = 'abcçdeəfgğhxıijklmnoöpqrsştuüvwyz'
        new_char = random.choice(az_chars)
        word = word[:pos] + new_char + word[pos:]
        return word, 'insertion'
    
    elif error_type == 'deletion' and len(word) > 1:
        pos = random.randint(0, len(word) - 1)
        word = word[:pos] + word[pos+1:]
        return word, 'deletion'
    
    elif error_type == 'transposition' and len(word) > 1:
        pos = random.randint(0, len(word) - 2)
        word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
        return word, 'transposition'
    
    return word, "correct"

# Spell checker that uses weighted edit distance
class WeightedSpellChecker:
    
    def __init__(self, vocabulary, max_distance=2.0):
        self.vocabulary = vocabulary
        self.max_distance = max_distance
        self.wed = WeightedEditDistance()
        self.confusion_matrix = defaultdict(int)
    
    def find_correction(self, word):
        # Find the best spelling correction for a word
        word = az_lowercase(word)
        
        # Check if it's already correct
        if word in self.vocabulary:
            return word, 0.0, "exact_match"
        
        # Look for similar words
        candidates = []
        max_length_diff = int(self.max_distance)
        
        # Only check words with similar length to save time
        candidate_list = [
            candidate for candidate in self.vocabulary
            if abs(len(candidate) - len(word)) <= max_length_diff
        ]
        
        # Don't check too many words or it takes forever
        if len(candidate_list) > 500:
            candidate_list = candidate_list[:500]
        
        for candidate in candidate_list:
            distance = self.wed.compute_distance(word, candidate)
            
            if distance <= self.max_distance:
                candidates.append((candidate, distance))
                # Stop early if we find a really good match
                if distance < 0.5:
                    break
        
        if not candidates:
            return None, float('inf'), "no_match"
        
        # Return the best match
        candidates.sort(key=lambda x: x[1])
        best_word, best_distance = candidates[0]
        
        return best_word, best_distance, "corrected"
    
    def update_confusion_matrix(self, actual_error, predicted_error):
        """
        Update confusion matrix for error type classification.
        """
        self.confusion_matrix[(actual_error, predicted_error)] += 1

# Generate Test Cases
print("Generating test cases with spelling errors...")

# Select random words from vocabulary for testing
test_words = random.sample(list(vocab_set), min(500, len(vocab_set)))  # Reduced from 1000
test_cases = []

for word in test_words:
    # Generate error version
    error_word, error_type = generate_spelling_errors(word, error_rate=0.5)
    test_cases.append({
        "original": word,
        "error_word": error_word,
        "error_type": error_type
    })

print(f"Generated {len(test_cases)} test cases")

# Run Spell Checking with Confusion Matrix
print("Running spell checking tests...")

checker = WeightedSpellChecker(vocab_set, max_distance=2.5)
results = []
correct_predictions = 0
total_predictions = 0

error_type_stats = defaultdict(int)
correction_stats = defaultdict(int)

for case in test_cases:
    original = case["original"]
    error_word = case["error_word"]
    actual_error_type = case["error_type"]
    
    # Get correction suggestion
    correction, distance, prediction_type = checker.find_correction(error_word)
    
    # Determine if correction was successful
    is_correct = (correction == original)
    
    if actual_error_type != "correct":  # Only count actual errors
        total_predictions += 1
        if is_correct:
            correct_predictions += 1
    
    # Update statistics
    error_type_stats[actual_error_type] += 1
    correction_stats[prediction_type] += 1
    
    # Update confusion matrix for error types
    predicted_error_type = "correct" if is_correct else "incorrect"
    checker.update_confusion_matrix(actual_error_type, predicted_error_type)
    
    # Store detailed result
    result = {
        "original_word": original,
        "error_word": error_word,
        "actual_error_type": actual_error_type,
        "predicted_word": correction,
        "weighted_distance": distance,
        "prediction_type": prediction_type,
        "is_correct": is_correct
    }
    results.append(result)

# Calculate Performance Metrics
accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

print(f"\n=== SPELL CHECKER PERFORMANCE ===")
print(f"Total test cases: {len(test_cases)}")
print(f"Cases with actual errors: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")

print(f"\n=== ERROR TYPE DISTRIBUTION ===")
for error_type, count in error_type_stats.items():
    percentage = (count / len(test_cases)) * 100
    print(f"{error_type}: {count} ({percentage:.1f}%)")

print(f"\n=== PREDICTION TYPE DISTRIBUTION ===")
for pred_type, count in correction_stats.items():
    percentage = (count / len(test_cases)) * 100
    print(f"{pred_type}: {count} ({percentage:.1f}%)")

# Save Results
print(f"\nSaving detailed results to {OUTPUT_FILE}...")
output_data = {
    "metadata": {
        "total_words_processed": len(all_words),
        "vocabulary_size": len(vocab_set),
        "test_cases": len(test_cases),
        "accuracy": accuracy,
        "max_weighted_distance": checker.max_distance
    },
    "error_type_stats": dict(error_type_stats),
    "correction_stats": dict(correction_stats),
    "test_results": results[:100]  # Save first 100 for brevity
}

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

# Save Confusion Matrix
print(f"Saving confusion matrix to {CONFUSION_MATRIX_FILE}...")
# Convert tuple keys to string keys for JSON compatibility
confusion_matrix_str_keys = {}
for (actual, predicted), count in checker.confusion_matrix.items():
    key = f"{actual}->{predicted}"
    confusion_matrix_str_keys[key] = count

confusion_data = {
    "confusion_matrix": confusion_matrix_str_keys,
    "matrix_explanation": {
        "format": "Key: actual_error_type->predicted_outcome",
        "values": "Count of occurrences"
    }
}

with open(CONFUSION_MATRIX_FILE, 'w', encoding='utf-8') as f:
    json.dump(confusion_data, f, ensure_ascii=False, indent=2)

# Generate Character-Level Confusion Matrix
print(f"Generating character confusion matrix...")
char_confusion = build_character_confusion_matrix(test_cases)

# Convert to JSON-compatible format
char_confusion_str_keys = {}
for (error_char, correct_char), count in char_confusion.items():
    key = f"{error_char}->{correct_char}"
    char_confusion_str_keys[key] = count

char_confusion_data = {
    "character_confusion_matrix": char_confusion_str_keys,
    "matrix_explanation": {
        "format": "Key: error_character->correct_character",
        "values": "Count of substitution occurrences"
    },
    "total_substitutions": sum(char_confusion.values()),
    "unique_character_pairs": len(char_confusion)
}

print(f"Saving character confusion matrix to {CHARACTER_CONFUSION_FILE}...")
with open(CHARACTER_CONFUSION_FILE, 'w', encoding='utf-8') as f:
    json.dump(char_confusion_data, f, ensure_ascii=False, indent=2)

# Display Character Confusion Matrix Summary
print(f"\n=== CHARACTER-LEVEL CONFUSION MATRIX ===")
print("Top 10 most common character substitutions:")
sorted_char_errors = sorted(char_confusion.items(), key=lambda x: x[1], reverse=True)
for i, ((error_char, correct_char), count) in enumerate(sorted_char_errors[:10], 1):
    print(f"{i:2d}. '{error_char}' -> '{correct_char}': {count} times")

print(f"\n=== CONFUSION MATRIX SUMMARY ===")
print("(Actual Error Type -> Prediction Outcome)")
for (actual, predicted), count in checker.confusion_matrix.items():
    print(f"{actual} -> {predicted}: {count}")

print(f"\nAll results saved successfully!")
print(f"- Detailed results: {OUTPUT_FILE}")
print(f"- Confusion matrix: {CONFUSION_MATRIX_FILE}")
print(f"- Character confusion matrix: {CHARACTER_CONFUSION_FILE}")