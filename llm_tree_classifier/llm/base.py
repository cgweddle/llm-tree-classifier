"""Base interface for LLM backends."""

from abc import ABC, abstractmethod
from typing import List


class LLMBackend(ABC):
    """Base class for LLM backends."""

    @abstractmethod
    def get_response(self, prompt: str, valid_responses: List[str]) -> str:
        """Get a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            valid_responses: List of valid response options

        Returns:
            The selected response from valid_responses
        """
        pass 