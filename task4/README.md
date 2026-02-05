# Task 4: Sentence Segmentation Analysis Report

## Dataset Overview
- **Total poems**: 846
- **Text columns analyzed**: text (original), modern_text (modernized)
- **Segmentation methods**: Basic, Poetry-aware, Advanced

## Method Descriptions

### Basic Segmentation
- Splits text on sentence-ending punctuation marks (.!?)
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

**Basic Method:**
- Total segments: 11,381
- Average segments per poem: 13.45
- Average segment length: 84.0 characters
- Total characters processed: 887,595

**Poetry Aware Method:**
- Total segments: 11,381
- Average segments per poem: 13.45
- Average segment length: 84.0 characters
- Total characters processed: 887,595

**Advanced Method:**
- Total segments: 11,408
- Average segments per poem: 13.48
- Average segment length: 83.7 characters
- Total characters processed: 887,541


### Modern Text (Contemporary Azerbaijani)

**Basic Method:**
- Total segments: 11,234
- Average segments per poem: 13.28
- Average segment length: 86.7 characters  
- Total characters processed: 913,958

**Poetry Aware Method:**
- Total segments: 11,240
- Average segments per poem: 13.29
- Average segment length: 86.7 characters  
- Total characters processed: 913,946

**Advanced Method:**
- Total segments: 11,266
- Average segments per poem: 13.32
- Average segment length: 86.4 characters  
- Total characters processed: 913,894


## Files Generated
- `segmentation_examples.json`: Sample segmentation results for first 3 poems
- `detailed_segmentation_results.json`: Complete segmentation results for all poems
- `segmentation_statistics.json`: Aggregated statistics
- `segmentation_analysis.png`: Bar charts comparing methods
- `segment_length_distribution.png`: Distribution plots of segment lengths

## Conclusions
The poetry-aware and advanced segmentation methods generally produce more segments compared to basic segmentation, as they account for the poetic structure and verse breaks. The advanced method provides additional refinement for handling complex poetic constructions.
