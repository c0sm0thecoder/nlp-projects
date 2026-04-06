"""
Task 1: Sentiment Analysis using BERT

Analyzes the nlptown/bert-base-multilingual-uncased-sentiment model to answer:
1. What are inputs and outputs of this model?
2. How many classes does it have?
3. What is the size of input?
4. Is model case sensitive (if yes how it affects accuracy)?
5. Is it possible to use this model for agglutinative languages (Azerbaijani)?
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"


def analyze_model_architecture(config, model, tokenizer) -> dict:
    """Extract model architecture details."""
    return {
        "model_name": MODEL_NAME,
        "base_model": "bert-base-multilingual-uncased",
        "num_classes": config.num_labels,
        "class_labels": config.id2label,
        "max_sequence_length": tokenizer.model_max_length,
        "vocab_size": tokenizer.vocab_size,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_hidden_layers": config.num_hidden_layers,
        "intermediate_size": config.intermediate_size,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def analyze_input_output(tokenizer, model, config) -> dict:
    """Demonstrate input and output format with examples."""
    sample_text = "This product is amazing! I love it so much."

    inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    return {
        "sample_text": sample_text,
        "input_format": {
            "description": "Text string tokenized into input_ids and attention_mask tensors",
            "input_keys": list(inputs.keys()),
            "input_ids_shape": list(inputs["input_ids"].shape),
            "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
        },
        "output_format": {
            "description": "Logits tensor for each class, convertible to probabilities via softmax",
            "logits_shape": list(outputs.logits.shape),
            "raw_logits": outputs.logits[0].tolist(),
            "probabilities": probabilities[0].tolist(),
            "predicted_class": predicted_class,
            "predicted_label": config.id2label[predicted_class],
            "confidence": probabilities[0][predicted_class].item(),
        },
    }


def analyze_case_sensitivity(tokenizer, model, config) -> dict:
    """Test whether the model is case sensitive."""
    test_pairs = [
        ("The food was delicious and the service was excellent!",
         "THE FOOD WAS DELICIOUS AND THE SERVICE WAS EXCELLENT!"),
        ("terrible experience, would not recommend",
         "TERRIBLE EXPERIENCE, WOULD NOT RECOMMEND"),
        ("It was okay, nothing special",
         "IT WAS OKAY, NOTHING SPECIAL"),
        ("I hate this product, worst purchase ever",
         "I HATE THIS PRODUCT, WORST PURCHASE EVER"),
        ("Best thing I have ever bought!",
         "BEST THING I HAVE EVER BOUGHT!"),
    ]

    results = []
    all_match = True

    for lower_text, upper_text in test_pairs:
        inputs_lower = tokenizer(lower_text, return_tensors="pt", truncation=True)
        inputs_upper = tokenizer(upper_text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            out_lower = model(**inputs_lower)
            out_upper = model(**inputs_upper)

        probs_lower = F.softmax(out_lower.logits, dim=-1)
        probs_upper = F.softmax(out_upper.logits, dim=-1)

        pred_lower = torch.argmax(probs_lower, dim=-1).item()
        pred_upper = torch.argmax(probs_upper, dim=-1).item()

        tokens_lower = tokenizer.convert_ids_to_tokens(inputs_lower["input_ids"][0])
        tokens_upper = tokenizer.convert_ids_to_tokens(inputs_upper["input_ids"][0])

        match = pred_lower == pred_upper
        if not match:
            all_match = False

        results.append({
            "lower_text": lower_text,
            "upper_text": upper_text,
            "tokens_match": tokens_lower == tokens_upper,
            "lower_prediction": config.id2label[pred_lower],
            "upper_prediction": config.id2label[pred_upper],
            "lower_confidence": probs_lower[0][pred_lower].item(),
            "upper_confidence": probs_upper[0][pred_upper].item(),
            "predictions_match": match,
        })

    return {
        "is_uncased_model": "uncased" in MODEL_NAME,
        "tokenizer_lowercases": True,  # mBERT uncased lowercases all input
        "case_affects_accuracy": False,
        "explanation": "This is an UNCASED model - all text is lowercased before tokenization, so case differences do not affect predictions.",
        "test_results": results,
        "all_predictions_match": all_match,
    }


def analyze_azerbaijani_support(tokenizer, model, config) -> dict:
    """Test model performance on Azerbaijani (agglutinative language)."""
    # Azerbaijani-English pairs with expected sentiment
    test_pairs = [
        ("Bu məhsul əladır! Çox xoşuma gəldi.",
         "This product is great! I liked it very much.", "positive"),
        ("Pis keyfiyyət, heç xoşuma gəlmədi.",
         "Bad quality, I didn't like it at all.", "negative"),
        ("Normal məhsuldur, pis deyil.",
         "It's a normal product, not bad.", "neutral"),
        ("Ən yaxşı alış-verişim! Tövsiyə edirəm.",
         "Best purchase! I recommend it.", "positive"),
        ("Çox pisdir, almayın.",
         "Very bad, don't buy it.", "negative"),
    ]

    comparison_results = []

    for az_text, en_text, expected in test_pairs:
        # Azerbaijani
        inputs_az = tokenizer(az_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out_az = model(**inputs_az)
        probs_az = F.softmax(out_az.logits, dim=-1)
        pred_az = torch.argmax(probs_az, dim=-1).item()

        # English
        inputs_en = tokenizer(en_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out_en = model(**inputs_en)
        probs_en = F.softmax(out_en.logits, dim=-1)
        pred_en = torch.argmax(probs_en, dim=-1).item()

        comparison_results.append({
            "azerbaijani_text": az_text,
            "english_text": en_text,
            "expected_sentiment": expected,
            "azerbaijani_prediction": config.id2label[pred_az],
            "english_prediction": config.id2label[pred_en],
            "azerbaijani_confidence": probs_az[0][pred_az].item(),
            "english_confidence": probs_en[0][pred_en].item(),
            "azerbaijani_tokens": tokenizer.convert_ids_to_tokens(inputs_az["input_ids"][0]),
            "english_tokens": tokenizer.convert_ids_to_tokens(inputs_en["input_ids"][0]),
        })

    # Tokenization analysis for agglutinative words
    agglutinative_words = [
        ("məhsullarımızdan", "from our products"),
        ("keyfiyyətsizlikləri", "their lack of quality"),
        ("xoşbəxtliklərimiz", "our happinesses"),
    ]

    tokenization_analysis = []
    for az_word, en_meaning in agglutinative_words:
        tokens = tokenizer.tokenize(az_word)
        tokenization_analysis.append({
            "azerbaijani_word": az_word,
            "english_meaning": en_meaning,
            "subword_tokens": tokens,
            "num_subwords": len(tokens),
        })

    return {
        "azerbaijani_in_mbert_vocab": True,
        "model_finetuned_languages": ["English", "Dutch", "German", "French", "Spanish", "Italian"],
        "azerbaijani_is_finetuned_language": False,
        "is_agglutinative": True,
        "agglutinative_challenge": "Azerbaijani words with multiple suffixes get heavily fragmented into subwords, losing morphological information.",
        "comparison_results": comparison_results,
        "tokenization_analysis": tokenization_analysis,
        "recommendation": "For better Azerbaijani sentiment analysis, fine-tune the model on Azerbaijani product reviews or use a model specifically trained for Turkic languages.",
        "conclusion": "The model CAN process Azerbaijani text since it's in mBERT vocabulary, but performance is degraded compared to fine-tuned languages due to: (1) no Azerbaijani training data, (2) heavy subword fragmentation of agglutinative words.",
    }


def generate_summary(architecture, case_sensitivity, azerbaijani) -> dict:
    """Generate a summary of all findings."""
    return {
        "question_1_inputs_outputs": {
            "inputs": "Text string, tokenized into input_ids (token indices) and attention_mask (1 for real tokens, 0 for padding)",
            "outputs": "Logits tensor of shape (batch_size, num_classes), converted to probabilities via softmax",
        },
        "question_2_num_classes": {
            "answer": architecture["num_classes"],
            "description": "5 classes representing star ratings from 1-star (very negative) to 5-stars (very positive)",
        },
        "question_3_input_size": {
            "max_sequence_length": architecture["max_sequence_length"],
            "vocab_size": architecture["vocab_size"],
            "note": "Texts longer than 512 tokens are truncated",
        },
        "question_4_case_sensitivity": {
            "is_case_sensitive": False,
            "explanation": case_sensitivity["explanation"],
            "accuracy_impact": "None - case differences do not affect model predictions",
        },
        "question_5_azerbaijani_support": {
            "is_possible": True,
            "quality": "Degraded compared to fine-tuned languages",
            "reasons": [
                "Azerbaijani is in mBERT vocabulary but not in fine-tuning data",
                "Agglutinative morphology causes heavy subword fragmentation",
                "Model lacks understanding of Azerbaijani sentiment expressions",
            ],
            "recommendation": azerbaijani["recommendation"],
        },
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded successfully!\n")

    # Run all analyses
    print("Analyzing model architecture...")
    architecture = analyze_model_architecture(config, model, tokenizer)

    print("Analyzing input/output format...")
    input_output = analyze_input_output(tokenizer, model, config)

    print("Analyzing case sensitivity...")
    case_sensitivity = analyze_case_sensitivity(tokenizer, model, config)

    print("Analyzing Azerbaijani language support...")
    azerbaijani = analyze_azerbaijani_support(tokenizer, model, config)

    print("Generating summary...")
    summary = generate_summary(architecture, case_sensitivity, azerbaijani)

    # Compile all results
    results = {
        "model_architecture": architecture,
        "input_output_analysis": input_output,
        "case_sensitivity_analysis": case_sensitivity,
        "azerbaijani_analysis": azerbaijani,
        "summary": summary,
    }

    # Save results
    results_path = out_dir / "sentiment_analysis_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"\nResults saved to: {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n1. INPUTS/OUTPUTS:")
    print(f"   Input: {summary['question_1_inputs_outputs']['inputs']}")
    print(f"   Output: {summary['question_1_inputs_outputs']['outputs']}")
    print(f"\n2. NUMBER OF CLASSES: {summary['question_2_num_classes']['answer']}")
    print(f"   {summary['question_2_num_classes']['description']}")
    print(f"\n3. INPUT SIZE:")
    print(f"   Max sequence length: {summary['question_3_input_size']['max_sequence_length']} tokens")
    print(f"   Vocabulary size: {summary['question_3_input_size']['vocab_size']}")
    print(f"\n4. CASE SENSITIVITY: {summary['question_4_case_sensitivity']['is_case_sensitive']}")
    print(f"   {summary['question_4_case_sensitivity']['explanation']}")
    print(f"\n5. AZERBAIJANI SUPPORT: {summary['question_5_azerbaijani_support']['is_possible']}")
    print(f"   Quality: {summary['question_5_azerbaijani_support']['quality']}")
    for reason in summary['question_5_azerbaijani_support']['reasons']:
        print(f"   - {reason}")


if __name__ == "__main__":
    main()
