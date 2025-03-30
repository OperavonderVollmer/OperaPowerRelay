# OperaPowerRelay

**OperaPowerRelay** is a collection of powerful utilities designed to simplify and automate various tasks. It provides tools for text manipulation, system information gathering, and additional utility functions to enhance workflow efficiency.

## Features

- **Text Utilities:** Functions for string manipulation, cleaning, and comparison.
- **System Tools:** Helpers for gathering system information and managing resources.
- **Other Utilities:** Additional functions to optimize programming tasks.

## Installation

### Prerequisites

- Python 3.x
- Required dependencies (install using pip):
  ```sh
  pip install inflect jellyfish metaphone nltk psutil pyperclip rapidfuzz sumy word2number
  ```

### Manual Installation

1. Clone or download the repository.
2. Navigate to the directory containing `setup.py`:
   ```sh
   cd /path/to/OperaPowerRelay
   ```
3. Install the package in **editable mode**:
   ```sh
   pip install -e .
   ```

### Installing via pip
```sh
pip install git+https://github.com/OperavonderVollmer/OperaPowerRelay.git@main
```

## Usage

### Importing as a Module

Once installed, you can import **OperaPowerRelay** into your Python project and use its utilities.

#### Example:
```python
from OperaPowerRelay import opr  # Importing the opr module

# Example usage of a function within the opr module
result = opr.some_function()
print(result)
```

## Error Handling

- If a required module is missing, an error is displayed.
- Ensure dependencies are installed correctly to avoid runtime issues.

## License

This project is open-source and available for modification and improvement.

