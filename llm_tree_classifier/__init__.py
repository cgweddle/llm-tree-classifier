"""LLM Tree Classifier package."""

import logging
from typing import List

from llm_tree_classifier.classifier import TreeClassifier, load_from_yaml
from llm_tree_classifier.exceptions import (
    LLMTreeClassifierError,
    TreeNotFoundError,
    InvalidTreeConfigError,
    LLMError,
)
from llm_tree_classifier.llm import LlamaCppBackend
from llm_tree_classifier.tree import DecisionNode, DecisionTree

__version__ = "0.1.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

__all__: List[str] = [
    "TreeClassifier",
    "load_from_yaml",
    "LlamaCppBackend",
    "DecisionNode",
    "DecisionTree",
    "LLMTreeClassifierError",
    "TreeNotFoundError",
    "InvalidTreeConfigError",
    "LLMError",
] 