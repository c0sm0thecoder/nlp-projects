# Project 5: RAG System

## Objective

In this project, you will design, implement, and evaluate a complete RAG system that can answer user questions based on a custom knowledge base (e.g., course materials, legal documents, FAQs, or company data).

The goal is to move beyond theory and build a working end-to-end system.

## Task 1. Data Preparation and Fundamentals (30%)

- Select a domain (education, legal, FAQ, etc.)
- Collect dataset (minimum 50 documents or 10,000+ words)
- Preprocess data:
  - Cleaning
  - Chunking (200–500 tokens)
- Briefly describe RAG components (Retriever, Generator, Knowledge Base)

## Task 2. RAG System Implementation (50%)

- Generate embeddings (Sentence-BERT or similar)
- Store in vector database (FAISS / ChromaDB)
- Implement retriever (top-k search)
- Integrate LLM (GPT / LLaMA / Mistral)
- Build full pipeline:
  - Query → Retrieval → Context → Answer
- Evaluate and compare with baseline (LLM without retrieval)

## Task 3. Report (20%)
