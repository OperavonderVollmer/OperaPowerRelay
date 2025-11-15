from setuptools import setup, find_packages

setup(
    name="OperaPowerRelay",
    version="1.2.0",
    packages=find_packages(),
    install_requires=[
        "inflect",
        "jellyfish",
        "metaphone",
        "nltk",
        "psutil",
        "pyperclip",
        "rapidfuzz",
        "sumy",
        "word2number",
        "winotify",
    ],
    python_requires=">=3.7",
    author="Opera von der Vollmer",
    description="A collection of useful utilities for various tasks.",
    url="https://github.com/OperavonderVollmer/OperaPowerRelay", 
    license="MIT",
)
