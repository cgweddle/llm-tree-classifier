"""Custom exceptions for the LLM Tree Classifier."""


class LLMTreeClassifierError(Exception):
    """Base exception for all LLM Tree Classifier errors."""


class TreeNotFoundError(LLMTreeClassifierError):
    """Raised when a requested tree is not found."""


class InvalidTreeConfigError(LLMTreeClassifierError):
    """Raised when tree configuration is invalid."""


class LLMError(LLMTreeClassifierError):
    """Raised when there is an error with the LLM backend.""" 