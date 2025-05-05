"""Decision tree implementation."""

from typing import Any, Dict, List, Optional

from llm_tree_classifier.llm import LLMBackend


class DecisionNode:
    """A node in the decision tree."""

    def __init__(
        self,
        question: Optional[str] = None,
        label: Optional[str] = None,
        options: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a decision node.

        Args:
            question: The question to ask (None for leaf nodes)
            label: The classification label (None for decision nodes)
            options: List of option dictionaries with 'value' and 'next' keys
        """
        if question is None and label is None:
            raise ValueError("Node must have either a question or a label")
        if question is not None and label is not None:
            raise ValueError("Node cannot have both a question and a label")
        if question is not None and not options:
            raise ValueError("Decision nodes must have options")

        self.question = question
        self.label = label
        self.options = options or []

    def is_leaf(self) -> bool:
        """Check if this is a leaf node.

        Returns:
            True if this is a leaf node, False otherwise
        """
        return self.label is not None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DecisionNode":
        """Create a node from a configuration dictionary.

        Args:
            config: The node configuration

        Returns:
            A new DecisionNode instance
        """
        if "label" in config:
            return cls(label=config["label"])

        # Process options recursively
        options = []
        for option in config["options"]:
            next_node = cls.from_dict(option["next"])
            options.append({
                "value": option["value"],
                "next": next_node
            })

        return cls(
            question=config["question"],
            options=options,
        )


class DecisionTree:
    """A decision tree for classification."""

    def __init__(self, name: str, root: DecisionNode) -> None:
        """Initialize a decision tree.

        Args:
            name: The name of the tree
            root: The root node of the tree
        """
        self.name = name
        self.root = root

    def classify(self, text: str, llm: LLMBackend) -> bool:
        """Classify text using this tree.

        Args:
            text: The text to classify
            llm: The LLM backend to use for decisions

        Returns:
            True if the text matches this tree's classification, False otherwise
        """
        node = self.root
        while not node.is_leaf():
            # Get valid response options
            valid_responses = [opt["value"] for opt in node.options]
            
            # Get response from LLM
            response = llm.get_response(node.question, valid_responses)
            
            # Find matching option
            for option in node.options:
                if option["value"] == response:
                    node = option["next"]
                    break
            else:
                # If no match found, use first option as fallback
                node = node.options[0]["next"]

        return node.label == "yes" 