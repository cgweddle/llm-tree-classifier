"""Tests for the LLM backend."""

import pytest
from typing import List

from llm_tree_classifier.llm.llama_cpp import LlamaCppBackend
from llm_tree_classifier.exceptions import LLMError


def test_llama_cpp_initialization(mocker) -> None:
    """Test LLaMA.cpp backend initialization.

    Args:
        mocker: Pytest mocker fixture
    """
    # Mock llama_cpp.Llama
    mock_llama = mocker.patch("llama_cpp.Llama")
    mock_llama.return_value = mocker.Mock()

    # Initialize backend
    backend = LlamaCppBackend(
        model_path="test_model.bin",
        n_ctx=512,
        n_batch=512,
        n_threads=4,
        n_gpu_layers=0
    )

    # Verify initialization
    assert backend.model is not None
    mock_llama.assert_called_once()


def test_llama_cpp_initialization_error(mocker) -> None:
    """Test LLaMA.cpp backend initialization error.

    Args:
        mocker: Pytest mocker fixture
    """
    # Mock llama_cpp.Llama to raise an error
    mock_llama = mocker.patch("llama_cpp.Llama")
    mock_llama.side_effect = Exception("Test error")

    # Verify initialization raises LLMError
    with pytest.raises(LLMError):
        LlamaCppBackend(
            model_path="test_model.bin",
            n_ctx=512,
            n_batch=512,
            n_threads=4,
            n_gpu_layers=0
        )


def test_get_response(mocker) -> None:
    """Test getting response from LLaMA.cpp backend.

    Args:
        mocker: Pytest mocker fixture
    """
    # Mock llama_cpp.Llama
    mock_llama = mocker.patch("llama_cpp.Llama")
    mock_model = mocker.Mock()
    mock_model.create_completion.return_value = {
        "choices": [{"text": "yes"}]
    }
    mock_llama.return_value = mock_model

    # Initialize backend
    backend = LlamaCppBackend(
        model_path="test_model.bin",
        n_ctx=512,
        n_batch=512,
        n_threads=4,
        n_gpu_layers=0
    )

    # Test response
    response = backend.get_response(
        question="Is this a test?",
        text="This is a test",
        valid_options=["yes", "no"]
    )
    assert response == "yes"


def test_get_response_error(mocker) -> None:
    """Test error handling in get_response.

    Args:
        mocker: Pytest mocker fixture
    """
    # Mock llama_cpp.Llama
    mock_llama = mocker.patch("llama_cpp.Llama")
    mock_model = mocker.Mock()
    mock_model.create_completion.side_effect = Exception("Test error")
    mock_llama.return_value = mock_model

    # Initialize backend
    backend = LlamaCppBackend(
        model_path="test_model.bin",
        n_ctx=512,
        n_batch=512,
        n_threads=4,
        n_gpu_layers=0
    )

    # Test error handling
    with pytest.raises(LLMError):
        backend.get_response(
            question="Is this a test?",
            text="This is a test",
            valid_options=["yes", "no"]
        ) 