"""Tests for the classifier module."""

import pytest
from typing import List

from llm_tree_classifier import TreeNotFoundError, InvalidTreeConfigError
from llm_tree_classifier.classifier import TreeClassifier


def test_classify_with_single_tree(mock_llm, sample_config) -> None:
    """Test classifying text with a single tree.

    Args:
        mock_llm: Mock LLM backend
        sample_config: Sample tree configuration
    """
    classifier = TreeClassifier(sample_config, mock_llm)
    result = classifier.classify("test text")
    assert isinstance(result, list)
    assert "test_tree" in result


def test_classify_with_specific_tree(mock_llm, sample_config) -> None:
    """Test classifying text with a specific tree.

    Args:
        mock_llm: Mock LLM backend
        sample_config: Sample tree configuration
    """
    classifier = TreeClassifier(sample_config, mock_llm, tree_name="test_tree")
    result = classifier.classify("test text")
    assert isinstance(result, list)
    assert "test_tree" in result


def test_initialization_with_nonexistent_tree(mock_llm, sample_config) -> None:
    """Test initialization with nonexistent tree.

    Args:
        mock_llm: Mock LLM backend
        sample_config: Sample tree configuration
    """
    with pytest.raises(TreeNotFoundError):
        TreeClassifier(sample_config, mock_llm, tree_name="nonexistent_tree")


def test_initialization_with_multiple_trees(mock_llm) -> None:
    """Test initialization with multiple trees.

    Args:
        mock_llm: Mock LLM backend
    """
    config = {
        "trees": [
            {
                "name": "tree1",
                "root": {
                    "question": "Is this tree 1?",
                    "options": [
                        {
                            "value": "yes",
                            "next": {"label": "yes"}
                        }
                    ]
                }
            },
            {
                "name": "tree2",
                "root": {
                    "question": "Is this tree 2?",
                    "options": [
                        {
                            "value": "yes",
                            "next": {"label": "yes"}
                        }
                    ]
                }
            }
        ]
    }

    # Should raise error when no tree_name specified
    with pytest.raises(InvalidTreeConfigError):
        TreeClassifier(config, mock_llm)

    # Should work when tree_name specified
    classifier = TreeClassifier(config, mock_llm, tree_name="tree1")
    result = classifier.classify("test text")
    assert isinstance(result, list)
    assert "tree1" in result 