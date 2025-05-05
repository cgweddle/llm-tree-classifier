"""Command-line interface for the LLM Tree Classifier."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from llm_tree_classifier import (
    LLMError,
    LlamaCppBackend,
    TreeNotFoundError,
    InvalidTreeConfigError,
    TreeClassifier,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Classify text using LLM-based decision trees."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to LLaMA model file",
    )
    parser.add_argument(
        "--tree",
        type=str,
        help="Specific tree to use for classification (required if config contains multiple trees)",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to classify (if not provided, reads from stdin)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    """Set up logging configuration.

    Args:
        verbose: Whether to enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_input_text(text: Optional[str]) -> str:
    """Get input text from argument or stdin.

    Args:
        text: Optional text from command line

    Returns:
        Text to classify
    """
    if text is not None:
        return text

    print("Enter text to classify (Ctrl+D to finish):")
    return sys.stdin.read().strip()


def main() -> int:
    """Run the CLI.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_args()
    setup_logging(args.verbose)

    try:
        # Initialize LLM
        llm = LlamaCppBackend(
            model_path=str(args.model),
            n_ctx=2048,
            n_batch=512,
            n_threads=None,
            n_gpu_layers=0,
        )

        # Create classifier
        classifier = TreeClassifier(args.config, llm, tree_name=args.tree)

        # Get input text
        text = get_input_text(args.text)
        if not text:
            print("Error: No text provided", file=sys.stderr)
            return 1

        # Classify text
        classifications = classifier.classify(text)
        print("Classifications:", ", ".join(classifications) or "None")
        return 0

    except (InvalidTreeConfigError, TreeNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except LLMError as e:
        print(f"Error with LLM: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 