import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Load Results
with open('spell_check_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

with open('confusion_matrix.json', 'r', encoding='utf-8') as f:
    confusion_data = json.load(f)

# Create Visualizations
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Error Type Distribution
error_stats = results['error_type_stats']
ax1 = axes[0, 0]
ax1.bar(error_stats.keys(), error_stats.values(), color='skyblue', edgecolor='black')
ax1.set_title('Distribution of Error Types', fontsize=14, fontweight='bold')
ax1.set_xlabel('Error Type')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)
for i, (k, v) in enumerate(error_stats.items()):
    ax1.text(i, v + 5, str(v), ha='center', va='bottom')

# 2. Correction Type Distribution
correction_stats = results['correction_stats']
ax2 = axes[0, 1]
ax2.pie(correction_stats.values(), labels=correction_stats.keys(), autopct='%1.1f%%', 
        colors=['lightcoral', 'lightgreen', 'lightsalmon'])
ax2.set_title('Prediction Type Distribution', fontsize=14, fontweight='bold')

# 3. Confusion Matrix Heatmap
confusion_matrix = confusion_data['confusion_matrix']
# Parse confusion matrix data
matrix_data = {}
for key, value in confusion_matrix.items():
    actual, predicted = key.split('->')
    if actual not in matrix_data:
        matrix_data[actual] = {}
    matrix_data[actual][predicted] = value

# Convert to DataFrame for heatmap
all_labels = set()
for actual in matrix_data:
    all_labels.add(actual)
    for predicted in matrix_data[actual]:
        all_labels.add(predicted)
all_labels = sorted(list(all_labels))

heatmap_data = np.zeros((len(all_labels), len(all_labels)))
for i, actual in enumerate(all_labels):
    for j, predicted in enumerate(all_labels):
        if actual in matrix_data and predicted in matrix_data[actual]:
            heatmap_data[i][j] = matrix_data[actual][predicted]

ax3 = axes[1, 0]
sns.heatmap(heatmap_data, xticklabels=all_labels, yticklabels=all_labels, 
            annot=True, fmt='g', cmap='Blues', ax=ax3)
ax3.set_title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold')
ax3.set_xlabel('Predicted Outcome')
ax3.set_ylabel('Actual Error Type')

# 4. Performance Metrics Summary
ax4 = axes[1, 1]
metrics = {
    'Total Test Cases': results['metadata']['total_words_processed'],
    'Vocabulary Size': results['metadata']['vocabulary_size'],
    'Test Cases': results['metadata']['test_cases'],
    'Accuracy (%)': results['metadata']['accuracy']
}

y_pos = np.arange(len(metrics))
values = list(metrics.values())
bars = ax4.barh(y_pos, values, color='lightsteelblue', edgecolor='black')
ax4.set_yticks(y_pos)
ax4.set_yticklabels(list(metrics.keys()))
ax4.set_xlabel('Value')
ax4.set_title('Performance Metrics Summary', fontsize=14, fontweight='bold')
ax4.set_xscale('log')  # Use log scale due to large differences in values

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, values)):
    ax4.text(value * 1.1, bar.get_y() + bar.get_height()/2, 
             f'{value:.1f}' if value < 100 else f'{int(value):,}',
             va='center', ha='left')

plt.tight_layout()
plt.savefig('spell_check_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as 'spell_check_analysis.png'")

# Detailed Analysis
print("\n=== DETAILED ANALYSIS ===")
print(f"Total words processed: {results['metadata']['total_words_processed']:,}")
print(f"Vocabulary size: {results['metadata']['vocabulary_size']:,}")
print(f"Spell check accuracy: {results['metadata']['accuracy']:.2f}%")
print(f"Maximum weighted distance threshold: {results['metadata']['max_weighted_distance']}")

print("\n=== CONFUSION MATRIX ANALYSIS ===")
total_errors = sum(v for k, v in confusion_matrix.items() if 'correct->correct' not in k)
correct_corrections = sum(v for k, v in confusion_matrix.items() if k.endswith('->correct') and not k.startswith('correct'))
incorrect_corrections = sum(v for k, v in confusion_matrix.items() if k.endswith('->incorrect'))

print(f"Total spelling errors detected: {total_errors}")
print(f"Correctly fixed errors: {correct_corrections}")
print(f"Incorrectly handled errors: {incorrect_corrections}")

if total_errors > 0:
    precision = correct_corrections / (correct_corrections + incorrect_corrections) * 100
    print(f"Error correction precision: {precision:.2f}%")

# Sample Results
print("\n=== SAMPLE CORRECTIONS ===")
sample_results = results['test_results'][:10]  # Show first 10 results
for i, result in enumerate(sample_results, 1):
    original = result['original_word']
    error = result['error_word']  
    prediction = result['predicted_word']
    distance = result['weighted_distance']
    is_correct = result['is_correct']
    
    status = "✓" if is_correct else "✗"
    print(f"{i:2d}. {original} → {error} → {prediction} (dist: {distance:.2f}) {status}")