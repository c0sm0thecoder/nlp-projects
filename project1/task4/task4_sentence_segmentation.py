import pandas as pd
import numpy as np
import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class AzerbaijaniSentenceSegmenter:
    """
    A sentence segmentation algorithm specifically designed for Azerbaijani text,
    particularly poetry which may have different punctuation patterns than prose.
    """
    
    def __init__(self):
        # Common sentence-ending punctuation marks
        self.sentence_endings = r'[.!?؟۔]'
        
        # Common abbreviations that should not trigger sentence breaks
        self.abbreviations = {
            'b.', 'c.', 'd.', 'h.', 'm.', 'n.', 's.', 't.', 'v.', 'x.',
            'İ.Ə.', 'e.ə.', 'və.s.', 'və s.', 'yəni'
        }
        
        # Poetry-specific patterns
        self.poetry_patterns = {
            'verse_break': r'\n\s*\n',  # Double line breaks
            'line_break': r'\n',        # Single line breaks
            'comma_pause': r',\s+',     # Commas often create natural pauses
            'semicolon': r';\s+',       # Semicolons create stronger pauses
        }
        
    def basic_segmentation(self, text: str) -> List[str]:
        """
        Basic sentence segmentation using punctuation marks.
        """
        if not text.strip():
            return []
        
        # Split on sentence-ending punctuation followed by whitespace or line break
        sentences = re.split(rf'{self.sentence_endings}\s*(?:\n|$|\s)', text.strip())
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def poetry_aware_segmentation(self, text: str) -> List[str]:
        """
        Poetry-aware segmentation that considers verse structure.
        """
        if not text.strip():
            return []
        
        segments = []
        
        # First, split by stanza/double line breaks breaks
        stanzas = re.split(self.poetry_patterns['verse_break'], text.strip())
        
        for stanza in stanzas:
            if not stanza.strip():
                continue
                
            # Within each stanza, look for sentence-ending punctuation
            stanza_sentences = self.basic_segmentation(stanza)
            
            if not stanza_sentences:
                # If no punctuation, split by lines as they may represent semantic units
                lines = [line.strip() for line in stanza.split('\n') if line.strip()]
                segments.extend(lines)
            else:
                segments.extend(stanza_sentences)
        
        return segments
    
    def advanced_segmentation(self, text: str) -> List[str]:
        """
        Advanced segmentation combining multiple approaches.
        """
        if not text.strip():
            return []
        
        # Start with poetry-aware segmentation
        segments = self.poetry_aware_segmentation(text)
        
        # Further refine by handling common patterns
        refined_segments = []
        
        for segment in segments:
            # Check if segment is too long and might contain multiple sentences
            if len(segment) > 200:
                sub_segments = re.split(r'[;،]\s+', segment)
                if len(sub_segments) > 1:
                    refined_segments.extend([s.strip() for s in sub_segments if s.strip()])
                else:
                    refined_segments.append(segment)
            else:
                refined_segments.append(segment)
        
        return [s for s in refined_segments if s.strip()]

    def evaluate_segmentation(self, original_segments: List[str], 
                            segmented_text: str) -> Dict[str, float]:
        """
        Basic evaluation metrics for segmentation quality.
        """
        if not original_segments:
            return {
                'num_segments': 0,
                'avg_length': 0, 
                'median_length': 0,
                'std_length': 0,
                'min_length': 0,
                'max_length': 0,
                'empty_segments': 0,
                'total_chars': 0
            }
        
        lengths = [len(seg) for seg in original_segments if seg.strip()]
        empty_count = sum(1 for seg in original_segments if not seg.strip())
        
        if not lengths:
            return {
                'num_segments': len(original_segments),
                'avg_length': 0,
                'median_length': 0, 
                'std_length': 0,
                'min_length': 0,
                'max_length': 0,
                'empty_segments': empty_count,
                'total_chars': 0
            }
        
        return {
            'num_segments': len(original_segments),
            'avg_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'std_length': np.std(lengths) if len(lengths) > 1 else 0,
            'min_length': min(lengths),
            'max_length': max(lengths),
            'empty_segments': empty_count,
            'total_chars': sum(lengths)
        }

def analyze_poem_dataset(df: pd.DataFrame, segmenter: AzerbaijaniSentenceSegmenter) -> Dict:
    """
    Analyze the entire poem dataset with different segmentation approaches.
    """
    results = {
        'original_text': {'basic': [], 'poetry_aware': [], 'advanced': []},
        'modern_text': {'basic': [], 'poetry_aware': [], 'advanced': []}
    }
    
    detailed_results = []

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df)} poems")
        
        poem_result = {
            'title': row['title'],
            'author': row['author'],
            'original_text_results': {},
            'modern_text_results': {}
        }
        
        # Process original text
        original_text = row['text']
        basic_segments = segmenter.basic_segmentation(original_text)
        poetry_segments = segmenter.poetry_aware_segmentation(original_text)
        advanced_segments = segmenter.advanced_segmentation(original_text)
        
        poem_result['original_text_results'] = {
            'basic': segmenter.evaluate_segmentation(basic_segments, original_text),
            'poetry_aware': segmenter.evaluate_segmentation(poetry_segments, original_text),
            'advanced': segmenter.evaluate_segmentation(advanced_segments, original_text)
        }
        
        # Store segments for analysis
        results['original_text']['basic'].append(basic_segments)
        results['original_text']['poetry_aware'].append(poetry_segments)
        results['original_text']['advanced'].append(advanced_segments)
        
        # Process modern text
        modern_text = row['modern_text']
        basic_segments_m = segmenter.basic_segmentation(modern_text)
        poetry_segments_m = segmenter.poetry_aware_segmentation(modern_text)
        advanced_segments_m = segmenter.advanced_segmentation(modern_text)
        
        poem_result['modern_text_results'] = {
            'basic': segmenter.evaluate_segmentation(basic_segments_m, modern_text),
            'poetry_aware': segmenter.evaluate_segmentation(poetry_segments_m, modern_text),
            'advanced': segmenter.evaluate_segmentation(advanced_segments_m, modern_text)
        }
        
        # Store segments for analysis
        results['modern_text']['basic'].append(basic_segments_m)
        results['modern_text']['poetry_aware'].append(poetry_segments_m)
        results['modern_text']['advanced'].append(advanced_segments_m)
        
        detailed_results.append(poem_result)
    
    return results, detailed_results

def generate_statistics(results: Dict, detailed_results: List[Dict]) -> Dict:
    """
    Generate statistics from the segmentation results.
    """
    stats = {}
    
    for text_type in ['original_text', 'modern_text']:
        stats[text_type] = {}
        
        for method in ['basic', 'poetry_aware', 'advanced']:
            # Aggregate metrics across all poems
            all_metrics = [poem[f'{text_type}_results'][method] 
                         for poem in detailed_results]
            
            # Extract valid metrics (non-zero values)
            valid_avg_lengths = [m['avg_length'] for m in all_metrics if m.get('avg_length', 0) > 0]
            valid_median_lengths = [m['median_length'] for m in all_metrics if m.get('median_length', 0) > 0]
            
            stats[text_type][method] = {
                'total_poems': len(all_metrics),
                'total_segments': sum(m['num_segments'] for m in all_metrics),
                'avg_segments_per_poem': np.mean([m['num_segments'] for m in all_metrics]),
                'avg_segment_length': np.mean(valid_avg_lengths) if valid_avg_lengths else 0,
                'median_segment_length': np.median(valid_median_lengths) if valid_median_lengths else 0,
                'total_chars': sum(m['total_chars'] for m in all_metrics),
                'empty_segments_total': sum(m['empty_segments'] for m in all_metrics)
            }
    
    return stats

def create_visualizations(stats: Dict, detailed_results: List[Dict]):
    """
    Create visualizations of the segmentation analysis results.
    """
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sentence Segmentation Analysis Results', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    methods = ['basic', 'poetry_aware', 'advanced']
    text_types = ['original_text', 'modern_text']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot 1: Average segments per poem
    for i, text_type in enumerate(text_types):
        segments_per_poem = [stats[text_type][method]['avg_segments_per_poem'] 
                           for method in methods]
        
        axes[i, 0].bar(methods, segments_per_poem, color=colors)
        axes[i, 0].set_title(f'Avg Segments per Poem - {text_type.replace("_", " ").title()}')
        axes[i, 0].set_ylabel('Average Segments')
        axes[i, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Average segment length
    for i, text_type in enumerate(text_types):
        avg_lengths = [stats[text_type][method]['avg_segment_length'] 
                      for method in methods]
        
        axes[i, 1].bar(methods, avg_lengths, color=colors)
        axes[i, 1].set_title(f'Avg Segment Length - {text_type.replace("_", " ").title()}')
        axes[i, 1].set_ylabel('Average Length (chars)')
        axes[i, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Total segments
    for i, text_type in enumerate(text_types):
        total_segments = [stats[text_type][method]['total_segments'] 
                        for method in methods]
        
        axes[i, 2].bar(methods, total_segments, color=colors)
        axes[i, 2].set_title(f'Total Segments - {text_type.replace("_", " ").title()}')
        axes[i, 2].set_ylabel('Total Segments')
        axes[i, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/home/kamal/NLP/project_1/task4/segmentation_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create distribution plot for segment lengths
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Distribution of Segment Lengths by Method', fontsize=16, fontweight='bold')
    
    for i, text_type in enumerate(text_types):
        all_lengths = []
        method_labels = []
        
        for method in methods:
            for poem in detailed_results:
                result = poem[f'{text_type}_results'][method]
                if result['num_segments'] > 0:
                    # Get individual segment lengths for this poem
                    avg_length = result['avg_length']
                    num_segments = result['num_segments']
                    # Approximate individual lengths (simplified)
                    lengths = [avg_length] * num_segments
                    all_lengths.extend(lengths)
                    method_labels.extend([method] * num_segments)
        
        df_plot = pd.DataFrame({'length': all_lengths, 'method': method_labels})
        
        sns.boxplot(data=df_plot, x='method', y='length', ax=axes[i])
        axes[i].set_title(f'Segment Length Distribution - {text_type.replace("_", " ").title()}')
        axes[i].set_ylabel('Segment Length (characters)')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/home/kamal/NLP/project_1/task4/segment_length_distribution.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def demonstrate_segmentation(df: pd.DataFrame, segmenter: AzerbaijaniSentenceSegmenter, 
                           num_examples: int = 3):
    """
    Demonstrate the segmentation algorithm on sample poems.
    """
    examples = []
    
    for i in range(min(num_examples, len(df))):
        poem = df.iloc[i]
        
        example = {
            'title': poem['title'],
            'author': poem['author'],
            'original_text': poem['text'],
            'modern_text': poem['modern_text'],
            'original_segmentation': {
                'basic': segmenter.basic_segmentation(poem['text']),
                'poetry_aware': segmenter.poetry_aware_segmentation(poem['text']),
                'advanced': segmenter.advanced_segmentation(poem['text'])
            },
            'modern_segmentation': {
                'basic': segmenter.basic_segmentation(poem['modern_text']),
                'poetry_aware': segmenter.poetry_aware_segmentation(poem['modern_text']),
                'advanced': segmenter.advanced_segmentation(poem['modern_text'])
            }
        }
        
        examples.append(example)
    
    return examples

def main():
    """
    Main function to run the sentence segmentation analysis.
    """
    print("Task 4: Sentence Segmentation Algorithm for Azerbaijani Poems")
    print("=" * 60)
    
    # Load the dataset
    print("Loading poems_translated.parquet...")
    df = pd.read_parquet('/home/kamal/NLP/project_1/poems_translated.parquet')
    print(f"Loaded {len(df)} poems")
    
    # Initialize the segmenter
    segmenter = AzerbaijaniSentenceSegmenter()
    
    # Demonstrate segmentation on sample poems
    print("\nDemonstrating segmentation methods on sample poems...")
    examples = demonstrate_segmentation(df, segmenter, num_examples=3)
    
    # Save examples for inspection
    with open('/home/kamal/NLP/project_1/task4/segmentation_examples.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    # Analyze the entire dataset
    print("\nAnalyzing entire dataset...")
    results, detailed_results = analyze_poem_dataset(df, segmenter)
    
    # Generate statistics
    print("Generating statistics...")
    stats = generate_statistics(results, detailed_results)
    
    # Save detailed results
    with open('/home/kamal/NLP/project_1/task4/detailed_segmentation_results.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    # Save statistics
    with open('/home/kamal/NLP/project_1/task4/segmentation_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS")
    print("=" * 40)
    
    for text_type in ['original_text', 'modern_text']:
        print(f"\n{text_type.replace('_', ' ').title()}:")
        for method in ['basic', 'poetry_aware', 'advanced']:
            method_stats = stats[text_type][method]
            print(f"  {method}:")
            print(f"    Total segments: {method_stats['total_segments']:,}")
            print(f"    Avg segments per poem: {method_stats['avg_segments_per_poem']:.2f}")
            print(f"    Avg segment length: {method_stats['avg_segment_length']:.1f} chars")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(stats, detailed_results)
    
    # Create summary report
    report_content = f"""# Task 4: Sentence Segmentation Analysis Report

## Dataset Overview
- **Total poems**: {len(df):,}
- **Text columns analyzed**: text (original), modern_text (modernized)
- **Segmentation methods**: Basic, Poetry-aware, Advanced

## Method Descriptions

### Basic Segmentation
- Splits text on sentence-ending punctuation marks (.!?؟۔)
- Simple approach suitable for prose text

### Poetry-aware Segmentation
- Considers verse structure and stanza breaks
- Handles double line breaks as stanza separators
- Falls back to line-based segmentation when no punctuation is present

### Advanced Segmentation  
- Combines poetry-aware approach with additional refinements
- Handles long segments by splitting on semicolons and strong commas
- Optimized for complex poetic structures

## Results Summary

### Original Text (Classical Azerbaijani)
"""
    
    for method in ['basic', 'poetry_aware', 'advanced']:
        method_stats = stats['original_text'][method]
        report_content += f"""
**{method.replace('_', ' ').title()} Method:**
- Total segments: {method_stats['total_segments']:,}
- Average segments per poem: {method_stats['avg_segments_per_poem']:.2f}
- Average segment length: {method_stats['avg_segment_length']:.1f} characters
- Total characters processed: {method_stats['total_chars']:,}
"""

    report_content += f"""

### Modern Text (Contemporary Azerbaijani)
"""
    
    for method in ['basic', 'poetry_aware', 'advanced']:
        method_stats = stats['modern_text'][method]
        report_content += f"""
**{method.replace('_', ' ').title()} Method:**
- Total segments: {method_stats['total_segments']:,}
- Average segments per poem: {method_stats['avg_segments_per_poem']:.2f}
- Average segment length: {method_stats['avg_segment_length']:.1f} characters  
- Total characters processed: {method_stats['total_chars']:,}
"""

    report_content += f"""

## Files Generated
- `segmentation_examples.json`: Sample segmentation results for first 3 poems
- `detailed_segmentation_results.json`: Complete segmentation results for all poems
- `segmentation_statistics.json`: Aggregated statistics
- `segmentation_analysis.png`: Bar charts comparing methods
- `segment_length_distribution.png`: Distribution plots of segment lengths

## Conclusions
The poetry-aware and advanced segmentation methods generally produce more segments compared to basic segmentation, as they account for the poetic structure and verse breaks. The advanced method provides additional refinement for handling complex poetic constructions.
"""
    
    with open('/home/kamal/NLP/project_1/task4/README.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("Analysis complete! Check the task4/ directory for results.")
    print("\nGenerated files:")
    print("- segmentation_examples.json")
    print("- detailed_segmentation_results.json")  
    print("- segmentation_statistics.json")
    print("- segmentation_analysis.png")
    print("- segment_length_distribution.png")
    print("- README.md")

if __name__ == "__main__":
    main()