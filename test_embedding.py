#!/usr/bin/env python3
"""
Test script for the embed_texts function
"""
import json
from rag.rag import embed_texts

def test_embed_texts():
    with open(f"data/grammar_409f784f4d3fceba.json", "r") as f:
        scfg = json.load(f)

    texts = scfg["grammar_str"].split('\n')
    
    # Use a small, fast model for testing
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    print(f"Testing embed_texts with {len(texts)} texts...")
    print(f"Using model: {model_name}")
    print("\nInput texts:")
    for i, text in enumerate(texts):
        print(f"  {i+1}. {text}")
    
    try:
        # Run the function
        embeddings = embed_texts(texts, model_name)
        
        print(f"\nSuccess! Generated embeddings:")
        print(f"Number of embeddings: {len(embeddings)}")
        print(f"Embedding dimension: {len(embeddings[0])}")
        print(f"First embedding (first 5 values): {embeddings[0][:5]}")
        
        # Test with single text
        print("\nTesting with single text...")
        single_embedding = embed_texts(["Single test sentence"], model_name)
        print(f"Single embedding dimension: {len(single_embedding[0])}")
        
        return True
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        return False

if __name__ == "__main__":
    test_embed_texts()
