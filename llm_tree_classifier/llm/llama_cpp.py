"""LLaMA.cpp implementation for the LLM Tree Classifier."""

import logging
from typing import List, Optional

from llama_cpp import Llama

from llm_tree_classifier.exceptions import LLMError
from llm_tree_classifier.llm.base import LLMBackend

logger = logging.getLogger(__name__)


class LlamaCppBackend(LLMBackend):
    """LLaMA.cpp implementation of the LLM backend."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
    ) -> None:
        """Initialize the LLaMA.cpp backend.

        Args:
            model_path: Path to the LLaMA model file
            n_ctx: Context window size
            n_batch: Batch size for prompt processing
            n_threads: Number of threads to use (None for auto)
            n_gpu_layers: Number of layers to offload to GPU

        Raises:
            LLMError: If there is an error initializing the model
        """
        try:
            logger.info(f"Initializing LLaMA.cpp backend with model: {model_path}")
            self.model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
            )
            logger.info("LLaMA.cpp backend initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLaMA.cpp backend: {e}")
            raise LLMError(f"Failed to initialize LLaMA.cpp backend: {e}")

    def get_response(self, prompt: str, valid_responses: List[str]) -> str:
        """Get a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            valid_responses: List of valid response options

        Returns:
            The selected response from valid_responses

        Raises:
            LLMError: If there is an error getting a response from the LLM
        """
        try:
            # Create grammar for valid responses
            grammar = self._create_grammar(valid_responses)

            # Add system prompt
            full_prompt = (
                "You are a decision maker. Choose one of the following options:\n"
                f"{', '.join(valid_responses)}\n\n"
                f"Question: {prompt}\nAnswer:"
            )

            logger.debug(f"Sending prompt to LLM: {full_prompt}")
            logger.debug(f"Valid responses: {valid_responses}")

            # Get response with grammar constraint
            response = self.model(
                full_prompt,
                max_tokens=10,
                stop=["\n"],
                temperature=0.0,
                grammar=grammar,
            )

            # Extract and normalize response
            answer = response["choices"][0]["text"].strip().lower()
            logger.debug(f"Raw LLM response: {answer}")

            # Find the closest matching valid response
            for valid in valid_responses:
                if valid.lower() in answer or answer in valid.lower():
                    logger.info(f"Selected response: {valid}")
                    return valid

            # If no match found, return the first valid response as fallback
            logger.warning(
                f"No exact match found for response: {answer}. Using fallback: {valid_responses[0]}"
            )
            return valid_responses[0]

        except Exception as e:
            logger.error(f"Error getting response from LLM: {e}")
            raise LLMError(f"Failed to get response from LLM: {e}")

    def _create_grammar(self, valid_responses: List[str]) -> str:
        """Create a grammar string for valid responses.

        Args:
            valid_responses: List of valid response options

        Returns:
            Grammar string in GBNF format
        """
        # Create root rule
        grammar = "root ::= response\n"

        # Create response rule with all valid options
        response_rule = "response ::= "
        response_rule += " | ".join(f'"{resp}"' for resp in valid_responses)
        grammar += response_rule + "\n"

        logger.debug(f"Created grammar: {grammar}")
        return grammar 