"""Main classifier implementation."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from llm_tree_classifier.exceptions import TreeNotFoundError, InvalidTreeConfigError
from llm_tree_classifier.llm.base import LLMBackend
from llm_tree_classifier.tree import DecisionTree

logger = logging.getLogger(__name__)


class TreeClassifier:
    """Classifier that uses a decision tree with LLM-based decisions."""

    def __init__(
        self,
        config: Union[str, Path, Dict],
        llm: LLMBackend,
        tree_name: Optional[str] = None,
    ) -> None:
        """Initialize the classifier.

        Args:
            config: Path to YAML file or dictionary containing tree configuration
            llm: LLM backend to use for decisions
            tree_name: Name of the tree to use (required if config contains multiple trees)

        Raises:
            InvalidTreeConfigError: If configuration is invalid
            TreeNotFoundError: If specified tree is not found
        """
        self.llm = llm

        # Load configuration
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = yaml.safe_load(f)

        # Load trees from configuration
        trees = {}
        for tree_config in config.get("trees", []):
            tree = DecisionTree.from_dict(tree_config)
            trees[tree.name] = tree

        if not trees:
            raise InvalidTreeConfigError("No trees found in configuration")

        # Select tree
        if tree_name is not None:
            if tree_name not in trees:
                raise TreeNotFoundError(
                    f"Tree '{tree_name}' not found. Available trees: {list(trees.keys())}"
                )
            self.tree = trees[tree_name]
        elif len(trees) == 1:
            self.tree = next(iter(trees.values()))
        else:
            raise InvalidTreeConfigError(
                f"Multiple trees found in configuration. Please specify tree_name. Available trees: {list(trees.keys())}"
            )

        logger.info("Initialized TreeClassifier with tree '%s'", self.tree.name)

    def classify(self, text: str) -> List[str]:
        """Classify text using the tree.

        Args:
            text: Text to classify

        Returns:
            List of classification results
        """
        logger.info("Classifying text using tree '%s'", self.tree.name)
        return self.tree.classify(text, self.llm) 