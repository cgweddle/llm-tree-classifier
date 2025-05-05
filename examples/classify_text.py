#!/usr/bin/env python3
"""
Example script demonstrating how to use the LLM Tree Classifier.
"""

import logging
from pathlib import Path
from llm_tree_classifier import TreeClassifier, LlamaCppBackend

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Initialize the LLM backend
    model_path = Path("path/to/your/model.gguf")  # Replace with your model path
    llm = LlamaCppBackend(model_path)

    # Load the classifier with a specific tree
    config_path = Path(__file__).parent / "tree_config.yaml"
    classifier = TreeClassifier(config_path, llm, tree_name="sentiment")

    # Example texts to classify
    texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "The service was okay, nothing special but not bad either.",
        "This is the worst experience I've ever had. Terrible service and quality.",
    ]

    # Classify each text
    for text in texts:
        print(f"\nText: {text}")
        result = classifier.classify(text)
        print(f"Classification: {result}")

if __name__ == "__main__":
    main() 