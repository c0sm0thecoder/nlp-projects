"""
Task 2.1: BiDAF Implementation for Reading Comprehension

This script demonstrates the BiDAF model implementation:
- Model architecture overview
- Input/output format
- Forward pass demonstration
- Model statistics
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from bidaf_model import BiDAF, compute_loss


def get_model_architecture_info(model: BiDAF) -> dict:
    """Extract detailed model architecture information."""
    return {
        "name": "BiDAF (Bidirectional Attention Flow)",
        "paper": "Seo et al. (2017) - Bidirectional Attention Flow for Machine Comprehension",
        "arxiv": "https://arxiv.org/abs/1611.01603",
        "framework": "PyTorch",
        "layers": {
            "1_character_embedding": {
                "description": "CNN over character sequences to capture morphological features",
                "components": ["Character Embedding", "1D Convolution", "Max Pooling"],
                "output_dim": "char_num_filters (100)",
            },
            "2_word_embedding": {
                "description": "Pre-trained word embeddings (GloVe/Word2Vec)",
                "output_dim": "word_embed_dim (100)",
            },
            "3_highway_network": {
                "description": "Combines character and word embeddings with gating mechanism",
                "num_layers": 2,
                "formula": "y = g * H(x) + (1-g) * x, where g = sigmoid(W_g * x)",
            },
            "4_contextual_embedding": {
                "description": "BiLSTM to capture temporal interactions",
                "type": "Bidirectional LSTM",
                "output_dim": "hidden_dim * 2 (200)",
            },
            "5_attention_flow": {
                "description": "Bidirectional attention between context and query",
                "components": {
                    "similarity_matrix": "S(i,j) = w^T [h_i; u_j; h_i * u_j]",
                    "context_to_query": "For each context word, weighted sum of query words",
                    "query_to_context": "Most relevant context word for each position",
                },
                "output": "G = [H; U_tilde; H*U_tilde; H*H_tilde]",
                "output_dim": "hidden_dim * 8 (800)",
            },
            "6_modeling_layer": {
                "description": "Stacked BiLSTM to capture interactions among context words",
                "type": "2-layer Bidirectional LSTM",
                "output_dim": "hidden_dim * 2 (200)",
            },
            "7_output_layer": {
                "description": "Predicts start and end positions of answer span",
                "start_prediction": "softmax(W_start * [G; M])",
                "end_prediction": "softmax(W_end * [G; M2]) where M2 = BiLSTM(M)",
            },
        },
    }


def get_model_parameters(model: BiDAF) -> dict:
    """Count parameters in each component."""
    param_counts = {}

    component_mapping = {
        "word_embed": "Word Embedding",
        "char_embed": "Character Embedding",
        "highway": "Highway Network",
        "contextual": "Contextual Embedding (BiLSTM)",
        "attention": "Attention Flow Layer",
        "modeling": "Modeling Layer (BiLSTM)",
        "output": "Output Layer",
    }

    for name, module in model.named_children():
        display_name = component_mapping.get(name, name)
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        param_counts[display_name] = {
            "total": total,
            "trainable": trainable,
        }

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "by_component": param_counts,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }


def demonstrate_forward_pass(model: BiDAF) -> dict:
    """Demonstrate model input/output with example data."""
    model.eval()

    # Example dimensions
    batch_size = 2
    context_len = 50
    query_len = 15
    word_len = 16  # max characters per word

    # Create dummy inputs
    context_word_ids = torch.randint(1, 1000, (batch_size, context_len))
    context_char_ids = torch.randint(1, 100, (batch_size, context_len, word_len))
    query_word_ids = torch.randint(1, 1000, (batch_size, query_len))
    query_char_ids = torch.randint(1, 100, (batch_size, query_len, word_len))

    # Masks (1 for valid tokens)
    context_mask = torch.ones(batch_size, context_len)
    query_mask = torch.ones(batch_size, query_len)

    # Forward pass
    with torch.no_grad():
        start_logits, end_logits = model(
            context_word_ids=context_word_ids,
            context_char_ids=context_char_ids,
            query_word_ids=query_word_ids,
            query_char_ids=query_char_ids,
            context_mask=context_mask,
            query_mask=query_mask,
        )

        # Get predictions
        start_positions, end_positions = model.get_answer_span(start_logits, end_logits)

        # Get probabilities
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)

    return {
        "input_format": {
            "context_word_ids": {
                "shape": list(context_word_ids.shape),
                "description": "Word indices for context passage",
            },
            "context_char_ids": {
                "shape": list(context_char_ids.shape),
                "description": "Character indices for each word in context",
            },
            "query_word_ids": {
                "shape": list(query_word_ids.shape),
                "description": "Word indices for question",
            },
            "query_char_ids": {
                "shape": list(query_char_ids.shape),
                "description": "Character indices for each word in question",
            },
            "context_mask": {
                "shape": list(context_mask.shape),
                "description": "Mask for valid context tokens (1=valid, 0=padding)",
            },
            "query_mask": {
                "shape": list(query_mask.shape),
                "description": "Mask for valid query tokens (1=valid, 0=padding)",
            },
        },
        "output_format": {
            "start_logits": {
                "shape": list(start_logits.shape),
                "description": "Logits for answer start position",
                "example_values": start_logits[0, :5].tolist(),
            },
            "end_logits": {
                "shape": list(end_logits.shape),
                "description": "Logits for answer end position",
                "example_values": end_logits[0, :5].tolist(),
            },
        },
        "predictions": {
            "start_positions": start_positions.tolist(),
            "end_positions": end_positions.tolist(),
            "start_confidence": [start_probs[i, start_positions[i]].item() for i in range(batch_size)],
            "end_confidence": [end_probs[i, end_positions[i]].item() for i in range(batch_size)],
        },
    }


def demonstrate_loss_computation() -> dict:
    """Show how loss is computed during training."""
    batch_size = 2
    context_len = 50

    # Dummy logits and ground truth
    start_logits = torch.randn(batch_size, context_len)
    end_logits = torch.randn(batch_size, context_len)
    start_positions = torch.tensor([10, 25])  # Ground truth start indices
    end_positions = torch.tensor([15, 30])  # Ground truth end indices

    loss = compute_loss(start_logits, end_logits, start_positions, end_positions)

    return {
        "loss_function": "Cross Entropy",
        "formula": "loss = (CE(start_logits, start_pos) + CE(end_logits, end_pos)) / 2",
        "example_loss": loss.item(),
        "ground_truth_start": start_positions.tolist(),
        "ground_truth_end": end_positions.tolist(),
    }


def get_hyperparameters() -> dict:
    """Default hyperparameters for BiDAF."""
    return {
        "word_embed_dim": {
            "value": 100,
            "description": "Dimension of word embeddings (GloVe-100d recommended)",
        },
        "char_embed_dim": {
            "value": 8,
            "description": "Dimension of character embeddings",
        },
        "char_num_filters": {
            "value": 100,
            "description": "Number of CNN filters for character embedding",
        },
        "hidden_dim": {
            "value": 100,
            "description": "Hidden dimension for BiLSTM layers",
        },
        "num_highway_layers": {
            "value": 2,
            "description": "Number of highway network layers",
        },
        "num_modeling_layers": {
            "value": 2,
            "description": "Number of BiLSTM layers in modeling layer",
        },
        "dropout": {
            "value": 0.2,
            "description": "Dropout rate",
        },
        "learning_rate": {
            "value": 0.5,
            "description": "Initial learning rate (with Adadelta optimizer)",
        },
        "batch_size": {
            "value": 60,
            "description": "Training batch size",
        },
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Task 2.1: BiDAF Implementation")
    print("=" * 60)

    # Model configuration
    word_vocab_size = 10000
    char_vocab_size = 100

    print("\nInitializing BiDAF model...")
    model = BiDAF(
        word_vocab_size=word_vocab_size,
        char_vocab_size=char_vocab_size,
        word_embed_dim=100,
        char_embed_dim=8,
        char_num_filters=100,
        hidden_dim=100,
        num_highway_layers=2,
        num_modeling_layers=2,
        dropout=0.2,
    )
    print("Model initialized successfully!")

    # Collect all information
    print("\nCollecting model architecture information...")
    architecture = get_model_architecture_info(model)

    print("Counting model parameters...")
    parameters = get_model_parameters(model)

    print("Demonstrating forward pass...")
    forward_pass = demonstrate_forward_pass(model)

    print("Demonstrating loss computation...")
    loss_info = demonstrate_loss_computation()

    print("Getting hyperparameters...")
    hyperparameters = get_hyperparameters()

    # Compile results
    results = {
        "model_architecture": architecture,
        "model_parameters": parameters,
        "input_output_demonstration": forward_pass,
        "loss_computation": loss_info,
        "hyperparameters": hyperparameters,
        "summary": {
            "description": "BiDAF (Bidirectional Attention Flow) is a reading comprehension model that uses bidirectional attention to fuse information from context and query.",
            "key_innovation": "The attention flow layer computes attention in both directions (context-to-query and query-to-context) without early summarization.",
            "inputs": [
                "Question (query) - tokenized into word and character indices",
                "Context passage - tokenized into word and character indices",
            ],
            "outputs": [
                "Start position logits - probability distribution over context tokens",
                "End position logits - probability distribution over context tokens",
            ],
            "answer_extraction": "The answer span is extracted from context[start_pos:end_pos+1]",
            "total_parameters": parameters["total_parameters"],
        },
    }

    # Save results
    results_path = out_dir / "bidaf_implementation_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"\nResults saved to: {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n1. MODEL ARCHITECTURE:")
    print("   - Character Embedding (CNN)")
    print("   - Word Embedding (GloVe/Word2Vec)")
    print("   - Highway Network (2 layers)")
    print("   - Contextual Embedding (BiLSTM)")
    print("   - Attention Flow Layer (C2Q + Q2C)")
    print("   - Modeling Layer (2-layer BiLSTM)")
    print("   - Output Layer (Start/End prediction)")

    print("\n2. INPUTS:")
    print("   - context_word_ids: (batch, context_len)")
    print("   - context_char_ids: (batch, context_len, word_len)")
    print("   - query_word_ids: (batch, query_len)")
    print("   - query_char_ids: (batch, query_len, word_len)")

    print("\n3. OUTPUTS:")
    print("   - start_logits: (batch, context_len)")
    print("   - end_logits: (batch, context_len)")
    print("   Answer span = context[argmax(start):argmax(end)+1]")

    print(f"\n4. TOTAL PARAMETERS: {parameters['total_parameters']:,}")

    print("\n5. PARAMETER BREAKDOWN:")
    for component, counts in parameters["by_component"].items():
        print(f"   - {component}: {counts['total']:,}")


if __name__ == "__main__":
    main()
