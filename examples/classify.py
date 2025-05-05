"""Example usage of the LLM Tree Classifier with LLaMA.cpp."""

import yaml
from pathlib import Path

from llm_tree_classifier.classifier import load_from_yaml
from llm_tree_classifier.llm import LlamaCppBackend


def main() -> None:
    """Run the example classifier."""
    # Load configuration
    config_path = Path(__file__).parent / "tree_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize LLaMA.cpp backend
    # You'll need to download a model file and update this path
    model_path = "path/to/your/model.gguf"
    llm = LlamaCppBackend(
        model_path=model_path,
        n_ctx=2048,  # Context window size
        n_batch=512,  # Batch size for prompt processing
        n_threads=None,  # Auto-detect number of threads
        n_gpu_layers=0,  # No GPU acceleration
    )

    # Create classifier
    classifier = load_from_yaml(config, llm)

    # Example texts to classify
    texts = [
        "I absolutely love this new AI technology! It's amazing!",
        "The weather is nice today.",
        "This product is terrible and I want my money back!",
        "The Lakers won their game last night.",
    ]

    # Classify each text using all trees
    print("\nClassifying with all trees:")
    for text in texts:
        print(f"\nText: {text}")
        classifications = classifier.classify(text)
        print(f"Classifications: {classifications}")

    # Classify using specific trees
    print("\nClassifying with specific trees:")
    for text in texts:
        print(f"\nText: {text}")
        # Try sentiment tree
        try:
            sentiment = classifier.classify(text, "sentiment")
            print(f"Sentiment classification: {sentiment}")
        except ValueError as e:
            print(f"Error with sentiment tree: {e}")

        # Try topic tree
        try:
            topic = classifier.classify(text, "topic")
            print(f"Topic classification: {topic}")
        except ValueError as e:
            print(f"Error with topic tree: {e}")


if __name__ == "__main__":
    main() 