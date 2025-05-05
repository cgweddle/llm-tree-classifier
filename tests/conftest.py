"""Pytest configuration and fixtures."""

import pytest
from typing import Dict, Any

from llm_tree_classifier import LlamaCppBackend, TreeClassifier


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample tree configuration for testing.

    Returns:
        Dictionary containing sample tree configuration
    """
    return {
        "trees": [
            {
                "name": "test_tree",
                "root": {
                    "question": "Is this a test?",
                    "options": [
                        {
                            "value": "yes",
                            "next": {
                                "label": "yes"
                            }
                        },
                        {
                            "value": "no",
                            "next": {
                                "label": "yes"
                            }
                        }
                    ]
                }
            }
        ]
    }


@pytest.fixture
def mock_llm(mocker):
    """Mock LLM backend for testing.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Mocked LLM backend
    """
    mock = mocker.Mock(spec=LlamaCppBackend)
    mock.get_response.return_value = "yes"
    return mock


@pytest.fixture
def classifier(mock_llm, sample_config):
    """Test classifier instance.

    Args:
        mock_llm: Mock LLM backend
        sample_config: Sample tree configuration

    Returns:
        Configured TreeClassifier instance
    """
    from llm_tree_classifier import load_from_yaml
    return load_from_yaml(sample_config, mock_llm) 