# OperaPowerRelay

**OperaPowerRelay** is a collection of powerful utilities designed to simplify and automate various tasks. With this package, you gain access to a range of useful functions that can enhance your workflow. It includes tools for text manipulation, system information gathering, and much more.

## Features
- **Text utilities**: Functions for string manipulation, cleaning, and comparison.
- **System tools**: Helpers for gathering system information and managing resources.
- **Other utilities**: Additional functions to enhance your day-to-day programming tasks.

## Installation

You can install **OperaPowerRelay** directly from your local development environment using `pip`:

### Manually
1. Clone or download the repository.

2. Navigate to the directory containing `setup.py`:

    ```bash
    cd /path/to/OperaPowerRelay
    ```

3. Install the package in **editable mode**:

    ```bash
    pip install -e .
    ```
### Using pip
pip install git+https://github.com/OperavonderVollmer/OperaPowerRelay.git@+the tag of the release version you want to install
```
pip install git+https://github.com/OperavonderVollmer/OperaPowerRelay.git@v1.1
```
Technically, 
```
pip install git+https://github.com/OperavonderVollmer/OperaPowerRelay.git
```
Also works but this installs the current commit which may or may not work :)

This will install **OperaPowerRelay** and allow you to import and use it in any Python project.

## Usage

Once installed, you can import **OperaPowerRelay** into your Python code and start using its utilities.

### Example: Importing and Using the Package

Here’s how to import the package:

```python
from OperaPowerRelay import opt  # Importing the opt module

# Example usage of a function within the opt module
result = opt.some_function()
print(result)
