import pandas as pd
import Levenshtein
import re
import json
from collections import Counter, defaultdict

FILE_PATH = '../poems_translated.parquet'
VOCAB_COLUMN = 'text'

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
    # Remove punctuation and numbers, keep only alphabet chars
    text = re.sub(r'[^\w\s-]', '', text)  # Keep hyphens for compound words
    return text.split()

def create_quick_substitution_map():
    """
    Create quick character substitution map for common Azerbaijani confusions.
    """
    return {
        'e': 'ə', 'ə': 'e',
        'u': 'ü', 'ü': 'u', 
        'o': 'ö', 'ö': 'o',
        'i': 'ı', 'ı': 'i',
        's': 'ş', 'ş': 's',
        'c': 'ç', 'ç': 'c',
        'g': 'ğ', 'ğ': 'g'
    }

def generate_quick_variants(word, substitution_map):
    """
    Quickly generate common variants for a word.
    """
    variants = set([word])
    
    # Single character substitutions
    for i, char in enumerate(word):
        if char in substitution_map:
            new_char = substitution_map[char]
            variant = word[:i] + new_char + word[i+1:]
            variants.add(variant)
    
    return variants

class FastAzerbaijaniSpellChecker:
    def __init__(self, vocab_set, vocab_counts):
        self.vocab_set = vocab_set
        self.vocab_counts = vocab_counts
        self.substitution_map = create_quick_substitution_map()
        
        # Pre-compute vocabulary by length for faster filtering
        self.vocab_by_length = defaultdict(set)
        for word in vocab_set:
            self.vocab_by_length[len(word)].add(word)
        
        # Pre-compute common variants lookup
        print("Pre-computing variant lookup table...")
        self.variant_lookup = {}
        for word in vocab_set:
            variants = generate_quick_variants(word, self.substitution_map)
            for variant in variants:
                if variant not in self.variant_lookup:
                    self.variant_lookup[variant] = []
                self.variant_lookup[variant].append(word)
        print(f"Variant lookup table created with {len(self.variant_lookup)} entries")
    
    def get_length_filtered_candidates(self, word, max_distance=3):
        """
        Get candidates filtered by length to reduce search space.
        """
        candidates = set()
        word_len = len(word)
        
        # Check words of similar length only
        for length in range(max(1, word_len - max_distance), 
                          word_len + max_distance + 1):
            candidates.update(self.vocab_by_length[length])
        
        return candidates
    
    def get_best_correction(self, word, aance=3):
        """
        Fast correction using multiple quick strategies.
        """
        original_word = word
        word_lower = az_lowercase(word)
        
        # Exact match
        if word_lower in self.vocab_set:
            return self.preserve_case(word_lower, original_word), 0
        
        # Quick variant lookup
        if word_lower in self.variant_lookup:
            candidates = self.variant_lookup[word_lower]
            best_variant = max(candidates, key=lambda w: self.vocab_counts.get(w, 0))
            return self.preserve_case(best_variant, original_word), 0.5
        
        # Fast Levenshtein on filtered candidates
        filtered_candidates = self.get_length_filtered_candidates(word_lower, max_distance)
        
        best_word = None
        min_distance = float('inf')
        
        for candidate in filtered_candidates:
            # Use fast built-in Levenshtein
            distance = Levenshtein.distance(word_lower, candidate)
            
            if distance < min_distance:
                min_distance = distance
                best_word = candidate
                
                # Early exit for very good matches
                if distance <= 1:
                    break
        
        if min_distance <= max_distance and best_word:
            return self.preserve_case(best_word, original_word), min_distance
        
        return None, float('inf')
    
    def preserve_case(self, corrected_word, original_word):
        """
        Preserve original capitalization pattern.
        """
        if not original_word or not corrected_word:
            return corrected_word
            
        if original_word[0].isupper() and corrected_word[0].islower():
            # Handle Azerbaijani-specific capitalization
            if corrected_word[0] == 'ı':
                return 'I' + corrected_word[1:]
            elif corrected_word[0] == 'i':
                return 'İ' + corrected_word[1:]
            else:
                return corrected_word[0].upper() + corrected_word[1:]
        
        return corrected_word

print("Loading Parquet file...")
try:
    df = pd.read_parquet(FILE_PATH)
    print(f"Loaded {len(df)} rows.")
except FileNotFoundError:
    print("File not found")
    exit(0)

# Build the Frequency Dictionary
print("Building vocabulary...")
all_words = []

for line in df[VOCAB_COLUMN]:
    tokens = preprocess_text(line)
    all_words.extend(tokens)

vocab_counts = Counter(all_words)
vocab_set = set(vocab_counts.keys())

print(f"Vocabulary built. Total unique words: {len(vocab_set)}")

# Initialize fast spell checker
print("Initializing fast spell checker...")
spell_checker = FastAzerbaijaniSpellChecker(vocab_set, vocab_counts)

print("\nTesting the Fast Spelling Checker")

input_file = 'spelling_tests.json'
output_file = 'test_results.json'

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find {input_file}")
    test_data = []

results = []
correct_count = 0

print(f"Processing {len(test_data)} test cases...")
for i, case in enumerate(test_data):
    if i % 50 == 0:
        print(f"  Processed {i}/{len(test_data)} cases...")
        
    input_word = case['input']
    expected_word = case['expected']
    error_type = case.get('type', 'unknown')
    
    # Run the fast checker
    suggestion, distance = spell_checker.get_best_correction(input_word, max_distance=4)
    
    # Determine success
    is_correct = (suggestion == expected_word)
    if is_correct:
        correct_count += 1
    
    result_entry = {
        "input_word": input_word,
        "expected_word": expected_word,
        "predicted_word": suggestion,
        "levenshtein_distance": distance,
        "status": "PASS" if is_correct else "FAIL",
        "error_type": error_type
    }
    
    results.append(result_entry)

# Save results
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# Print summary
if test_data:
    accuracy = (correct_count / len(test_data)) * 100
    print(f"\nFast Processing complete.")
    print(f"Total Tests: {len(test_data)}")
    print(f"Passed:      {correct_count}")
    print(f"Failed:      {len(test_data) - correct_count}")
    print(f"Accuracy:    {accuracy:.2f}%")
    print(f"Improvement: {accuracy - 12:.2f}% over baseline")
    print(f"Detailed results saved to '{output_file}'")
else:
    print("No test data found.")

# Error analysis
print("\nError Analysis:")
error_types = defaultdict(lambda: {'total': 0, 'correct': 0})

for result in results:
    error_type = result['error_type']
    error_types[error_type]['total'] += 1
    if result['status'] == 'PASS':
        error_types[error_type]['correct'] += 1

for error_type, stats in error_types.items():
    accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
    print(f"  {error_type}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")

print(f"\nResults saved to: {output_file}")