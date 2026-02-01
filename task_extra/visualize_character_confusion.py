import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict

# Load the character confusion data
try:
    with open('character_confusion_matrix.json', 'r', encoding='utf-8') as f:
        char_data = json.load(f)
    print("Character confusion matrix loaded successfully!")
except FileNotFoundError:
    print("Error: character_confusion_matrix.json not found!")
    print("Please run task_extra_spell_check.py first.")
    exit(1)

# Get the data
char_confusion = char_data['character_confusion_matrix']
total_substitutions = char_data['total_substitutions']

print(f"Total character substitutions found: {total_substitutions}")
print(f"Unique character pairs: {char_data['unique_character_pairs']}")

# Convert to list for easier processing
confusion_pairs = []
for key, count in char_confusion.items():
    error_char, correct_char = key.split('->')
    confusion_pairs.append((error_char, correct_char, count))

# Sort by how often it happens
confusion_pairs.sort(key=lambda x: x[2], reverse=True)

# Azerbaijani alphabet
az_alphabet = ['a', 'b', 'c', 'ç', 'd', 'e', 'ə', 'f', 'g', 'ğ', 'h', 'x', 'ı', 'i', 'j', 'k', 'q', 'l', 'm', 'n', 'o', 'ö', 'p', 'r', 's', 'ş', 't', 'u', 'ü', 'v', 'w', 'y', 'z']

# Find all characters that appear in mistakes
all_chars = set()
for error_char, correct_char, _ in confusion_pairs:
    all_chars.add(error_char)
    all_chars.add(correct_char)

# Sort them nicely
valid_chars = [char for char in az_alphabet if char in all_chars]
other_chars = [char for char in sorted(all_chars) if char not in az_alphabet]
all_sorted_chars = valid_chars + other_chars

print(f"Characters found in confusion matrix: {len(all_sorted_chars)}")
print(f"Characters: {all_sorted_chars}")

# Make the confusion matrix
matrix_size = len(all_sorted_chars)
confusion_matrix = np.zeros((matrix_size, matrix_size))

char_to_index = {char: i for i, char in enumerate(all_sorted_chars)}

# Fill in the numbers
for error_char, correct_char, count in confusion_pairs:
    if error_char in char_to_index and correct_char in char_to_index:
        error_idx = char_to_index[error_char]
        correct_idx = char_to_index[correct_char]
        confusion_matrix[error_idx][correct_idx] = count

# Create Visualizations
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# 1. Full Character Confusion Matrix Heatmap
ax1 = axes[0, 0]
sns.heatmap(confusion_matrix, 
            xticklabels=all_sorted_chars, 
            yticklabels=all_sorted_chars,
            annot=True, 
            fmt='g', 
            cmap='Reds', 
            ax=ax1,
            cbar_kws={'label': 'Substitution Count'})
ax1.set_title('Character-Level Confusion Matrix\n(Error Character → Correct Character)', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Correct Character')
ax1.set_ylabel('Error Character')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax1.get_yticklabels(), rotation=0)

# 2. Top Character Substitutions Bar Chart
ax2 = axes[0, 1]
top_15 = confusion_pairs[:15]
labels = [f"'{error}' → '{correct}'" for error, correct, _ in top_15]
counts = [count for _, _, count in top_15]

bars = ax2.barh(range(len(labels)), counts, color='lightcoral', edgecolor='black')
ax2.set_yticks(range(len(labels)))
ax2.set_yticklabels(labels)
ax2.set_xlabel('Number of Substitutions')
ax2.set_title('Top 15 Character Substitution Errors', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

# Add value labels on bars
for bar, count in zip(bars, counts):
    ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
             str(count), va='center', ha='left')

# 3. Vowel vs Consonant Substitution Analysis
vowels = {'a', 'e', 'ə', 'i', 'ı', 'o', 'ö', 'u', 'ü'}
vowel_to_vowel = 0
vowel_to_consonant = 0
consonant_to_vowel = 0
consonant_to_consonant = 0

for error_char, correct_char, count in confusion_pairs:
    error_is_vowel = error_char in vowels
    correct_is_vowel = correct_char in vowels
    
    if error_is_vowel and correct_is_vowel:
        vowel_to_vowel += count
    elif error_is_vowel and not correct_is_vowel:
        vowel_to_consonant += count
    elif not error_is_vowel and correct_is_vowel:
        consonant_to_vowel += count
    else:
        consonant_to_consonant += count

ax3 = axes[1, 0]
categories = ['Vowel→Vowel', 'Vowel→Consonant', 'Consonant→Vowel', 'Consonant→Consonant']
values = [vowel_to_vowel, vowel_to_consonant, consonant_to_vowel, consonant_to_consonant]
colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink']

bars = ax3.bar(categories, values, color=colors, edgecolor='black')
ax3.set_title('Character Substitution Patterns\n(Vowel vs Consonant)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Number of Substitutions')
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

# Add value labels on bars
for bar, value in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             str(value), ha='center', va='bottom')

# 4. Character Frequency Analysis
error_char_freq = defaultdict(int)
correct_char_freq = defaultdict(int)

for error_char, correct_char, count in confusion_pairs:
    error_char_freq[error_char] += count
    correct_char_freq[correct_char] += count

ax4 = axes[1, 1]
# Show top 10 most frequently confused characters (as errors)
top_error_chars = sorted(error_char_freq.items(), key=lambda x: x[1], reverse=True)[:10]
chars = [f"'{char}'" for char, _ in top_error_chars]
freqs = [freq for _, freq in top_error_chars]

bars = ax4.bar(chars, freqs, color='lightsteelblue', edgecolor='black')
ax4.set_title('Most Frequently Mistyped Characters', fontsize=14, fontweight='bold')
ax4.set_xlabel('Character')
ax4.set_ylabel('Total Error Frequency')
plt.setp(ax4.get_xticklabels(), rotation=0)

# Add value labels on bars
for bar, freq in zip(bars, freqs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
             str(freq), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('character_confusion_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print Detailed Analysis
print("\n" + "="*60)
print("CHARACTER-LEVEL CONFUSION MATRIX ANALYSIS")
print("="*60)

print(f"\nTOTAL STATISTICS:")
print(f"- Total character substitutions: {total_substitutions}")
print(f"- Unique character pairs: {len(confusion_pairs)}")
print(f"- Characters involved: {len(all_sorted_chars)}")

print(f"\nTOP 10 CHARACTER SUBSTITUTIONS:")
for i, (error_char, correct_char, count) in enumerate(confusion_pairs[:10], 1):
    percentage = (count / total_substitutions) * 100
    print(f"{i:2d}. '{error_char}' → '{correct_char}': {count:3d} times ({percentage:.1f}%)")

print(f"\nVOWEL vs CONSONANT ANALYSIS:")
total_vowel_errors = vowel_to_vowel + vowel_to_consonant
total_consonant_errors = consonant_to_vowel + consonant_to_consonant
print(f"- Vowel substitution errors: {total_vowel_errors} ({(total_vowel_errors/total_substitutions)*100:.1f}%)")
print(f"- Consonant substitution errors: {total_consonant_errors} ({(total_consonant_errors/total_substitutions)*100:.1f}%)")
print(f"- Vowel→Vowel: {vowel_to_vowel} ({(vowel_to_vowel/total_substitutions)*100:.1f}%)")
print(f"- Consonant→Consonant: {consonant_to_consonant} ({(consonant_to_consonant/total_substitutions)*100:.1f}%)")

print(f"\nMOST PROBLEMATIC CHARACTERS (frequently mistyped):")
for i, (char, freq) in enumerate(top_error_chars[:8], 1):
    percentage = (freq / total_substitutions) * 100
    print(f"{i}. '{char}': {freq} errors ({percentage:.1f}%)")

print(f"\nVisualization saved as 'character_confusion_analysis.png'")
print("="*60)