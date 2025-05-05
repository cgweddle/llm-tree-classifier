# LLM Tree Classifier

A Python package for classifying text using LLM-powered decision trees.

## Features

- Configurable decision trees via YAML
- Support for multiple discrete outputs at each node
- Mock LLM implementation for testing
- Easy integration with real LLM providers
- Extensible architecture

## Installation

### From Git Repository

Using pip:
```bash
pip install git+https://github.com/yourusername/llm-tree-classifier.git
```

Using conda:
```bash
conda install -c conda-forge git
pip install git+https://github.com/yourusername/llm-tree-classifier.git
```

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-tree-classifier.git
cd llm-tree-classifier
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Usage

### Basic Usage

```python
from llm_tree_classifier import TreeClassifier, LlamaCppBackend

# Initialize the LLM backend
llm = LlamaCppBackend("path/to/model.gguf")

# Load the classifier with a specific tree
classifier = TreeClassifier("tree_config.yaml", llm, tree_name="sentiment")

# Classify text
result = classifier.classify("This is a great product!")
print(result)  # Output: very_positive
```

### Command Line Interface

```bash
# Classify text using a specific tree
python -m llm_tree_classifier --config tree_config.yaml --model model.gguf --tree sentiment --text "This is a great product!"

# Classify text from stdin
echo "This is a great product!" | python -m llm_tree_classifier --config tree_config.yaml --model model.gguf --tree sentiment
```

### Tree Configuration

Define your decision trees in YAML format:

```yaml
trees:
  sentiment:
    root:
      question: "What is the sentiment of this text?"
      options:
        - value: "positive"
          next:
            label: "positive"
        - value: "negative"
          next:
            label: "negative"
        - value: "neutral"
          next:
            label: "neutral"
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=llm_tree_classifier

# Run type checking
mypy llm_tree_classifier/

# Run linting
ruff check .
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality. The hooks will run automatically on commit, but you can also run them manually:

```bash
pre-commit run --all-files
```

### Continuous Integration

The project uses GitHub Actions for continuous integration. The workflow:
- Runs tests
- Performs type checking
- Runs linting
- Uploads coverage reports

### CLI Usage

The package provides a command-line interface:

```bash
python -m llm_tree_classifier --config tree_config.yaml --model path/to/model.bin --text "Your text here"
```

Options:
- `--config`: Path to YAML configuration file (required)
- `--model`: Path to LLaMA model file (required)
- `--tree`: Name of specific tree to use (required if config contains multiple trees)
- `--text`: Text to classify (optional, can also be provided via stdin)
- `--verbose`: Enable verbose logging

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.