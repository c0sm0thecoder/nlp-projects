#!/usr/bin/env python3
"""
Web UI for testing the Azerbaijani Levenshtein spell checker.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import Levenshtein
import re
import json
from collections import Counter, defaultdict
import os

app = Flask(__name__)

# Import functions from the original file
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
    
    def get_best_correction(self, word, max_distance=3):
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

# Global spell checker instance
spell_checker = None

def initialize_spell_checker():
    """Initialize the spell checker with vocabulary from the dataset."""
    global spell_checker
    
    if spell_checker is not None:
        return
    
    FILE_PATH = '../poems_translated.parquet'
    VOCAB_COLUMN = 'text'
    
    print("Loading Parquet file...")
    try:
        df = pd.read_parquet(FILE_PATH)
        print(f"Loaded {len(df)} rows.")
    except FileNotFoundError:
        print("Error: poems_translated.parquet not found!")
        return False
    
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
    print("Spell checker initialized successfully!")
    return True

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_spelling():
    """Check spelling for a given word."""
    global spell_checker
    
    if spell_checker is None:
        return jsonify({
            'error': 'Spell checker not initialized. Please restart the server.'
        }), 500
    
    data = request.get_json()
    word = data.get('word', '').strip()
    max_distance = data.get('max_distance', 3)
    
    if not word:
        return jsonify({'error': 'No word provided'}), 400
    
    try:
        # Get correction
        correction, distance = spell_checker.get_best_correction(word, max_distance)
        
        # Check if word exists in vocabulary
        word_lower = az_lowercase(word)
        exists_in_vocab = word_lower in spell_checker.vocab_set
        word_frequency = spell_checker.vocab_counts.get(word_lower, 0)
        
        # Get some alternative suggestions if correction failed
        alternatives = []
        if correction is None or distance > max_distance:
            # Get a few close matches
            candidates = spell_checker.get_length_filtered_candidates(word_lower, max_distance + 1)
            candidate_distances = []
            
            for candidate in list(candidates)[:50]:  # Limit to first 50 for performance
                dist = Levenshtein.distance(word_lower, candidate)
                if dist <= max_distance + 1:
                    candidate_distances.append((candidate, dist, spell_checker.vocab_counts.get(candidate, 0)))
            
            # Sort by distance then by frequency
            candidate_distances.sort(key=lambda x: (x[1], -x[2]))
            alternatives = [{'word': c[0], 'distance': c[1], 'frequency': c[2]} 
                           for c in candidate_distances[:5]]
        
        result = {
            'input_word': word,
            'exists_in_vocab': exists_in_vocab,
            'word_frequency': word_frequency,
            'correction': correction,
            'distance': distance if correction else None,
            'alternatives': alternatives,
            'max_distance_used': max_distance
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error processing word: {str(e)}'}), 500

@app.route('/batch_check', methods=['POST'])
def batch_check():
    """Check spelling for multiple words."""
    global spell_checker
    
    if spell_checker is None:
        return jsonify({
            'error': 'Spell checker not initialized. Please restart the server.'
        }), 500
    
    data = request.get_json()
    text = data.get('text', '').strip()
    max_distance = data.get('max_distance', 3)
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Preprocess text to get individual words
        words = preprocess_text(text)
        results = []
        
        for word in words:
            if word:  # Skip empty words
                correction, distance = spell_checker.get_best_correction(word, max_distance)
                word_lower = az_lowercase(word)
                exists_in_vocab = word_lower in spell_checker.vocab_set
                word_frequency = spell_checker.vocab_counts.get(word_lower, 0)
                
                results.append({
                    'input_word': word,
                    'exists_in_vocab': exists_in_vocab,
                    'word_frequency': word_frequency,
                    'correction': correction,
                    'distance': distance if correction else None
                })
        
        return jsonify({
            'results': results,
            'total_words': len(results),
            'max_distance_used': max_distance
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing text: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Azerbaijani Spell Checker UI...")
    
    if initialize_spell_checker():
        print("Starting Flask server...")
        print("Open your browser and go to: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize spell checker. Please check if poems_translated.parquet exists.")