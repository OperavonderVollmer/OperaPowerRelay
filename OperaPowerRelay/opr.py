"""
    OperaPowerRelay
    Yippie!!!
"""

CURRENT_VERSION = "v1.1.5"

def get_version() -> str:
    """
        Every time, I have to change this a little bit lol
    """
    return CURRENT_VERSION

def opr_info() -> str:
    """
    Provide a brief description of the OperaPowerRelay utility.

    Returns
    -------
    str
        A message describing the purpose and design philosophy of the OperaPowerRelay.
        The toolkit is designed for lightweight, cross-project utility functions with 
        minimal load times due to on-demand imports.
    """

    message = (
        "OperaPowerRelay is a lightweight utility toolkit for functions that aren't worth defining "
        "in a single script but are useful across multiple projects. Imports are handled within "
        "each function to keep load times minimal."
    )
    return message

def find_best_match(target: str, options: list) -> str | None :
    from rapidfuzz import process
    """
    Find the best match for a target string in a list of options.

    Parameters
    ----------
    target : str
        The target string to find a match for.
    options : list
        A list of strings to search for a match.

    Returns
    -------
    str or None
        The best match, or None if no match is found.

    Notes
    -----
    The best match is determined by finding the string in options with the highest
    Levenshtein similarity to the target string. The similarity is calculated using
    the jellyfish library. If the similarity is below 90, no match is returned.
    """
    bestMatch = process.extractOne(target, options, score_cutoff=90)
    return bestMatch[0] if bestMatch else None

def get_phonetic_representation(word: str) -> str:
    """
    Return a phonetic representation of the given word.

    Parameters
    ----------
    word : str
        The word to get a phonetic representation of.

    Returns
    -------
    str
        A phonetic representation of the word. This is computed using the
        double metaphone algorithm, and if the result is empty, the soundex
        algorithm is used instead.
    """

    from metaphone import doublemetaphone
    import jellyfish  
    
    metaphone_primary, metaphone_secondary = doublemetaphone(word)
    return metaphone_primary or metaphone_secondary or jellyfish.soundex(word)

def find_best_phonetic_match(target: str, options: list) -> str | None:    


    """
    Find the best match in options by computing the Levenshtein similarity between
    the phonetic representation of the target and the phonetic representation of
    each option.

    Parameters
    ----------
    target : str
        The target string to find a match for.
    options : list[str]
        The list of strings to search for the best match.

    Returns
    -------
    str | None
        The best match found in options, or None if no match was found with a
        similarity of at least 70.
    """


    from rapidfuzz import process
    
    target_phonetic = get_phonetic_representation(target)
    phonetic_map = {option: get_phonetic_representation(option) for option in options}
    best_match = process.extractOne(target_phonetic, phonetic_map.values(), score_cutoff=70)

    if best_match:
        for option, phonetic in phonetic_map.items():
            if phonetic == best_match[0]:
                return option

    return None

def normalize_number(input: int | str) -> int:
    
    """
    Normalize a given input into an integer.

    Parameters
    ----------
    input : int | str
        The input to normalize. If it is an integer, it will be returned as is.
        If it is a string, it will be stripped of its whitespace and converted to
        lowercase. If the string is a number, it will be returned as an integer.
        If the string is a word representation of a number, it will be converted
        to an integer.

    Returns
    -------
    int
        The normalized integer.

    Raises
    ------
    ValueError
        If the input is not a valid number or word representation of a number.
    """
    from word2number import w2n
    import re
    import inflect

    if isinstance(input, int): return input

    input = input.strip().lower()

    if input.isdigit():  
        return int(input)
    
    p = inflect.engine()
    input = re.sub(r'\b\d+\b', lambda x: p.number_to_words(x.group(0)), input)

    try:
        return w2n.word_to_num(input)
    except ValueError as e:
        raise ValueError(f"Invalid number input: {input}") from e

def bubble_sort(arr: list[int]) -> list[int]:
    """
    Sorts the given list of integers in ascending order using the bubble sort algorithm.

    The bubble sort algorithm works by repeatedly swapping adjacent elements if they are in the wrong order.
    The algorithm stops when no more swaps occur, indicating that the list is sorted.

    Parameters
    ----------
    arr : list[int]
        The list of integers to be sorted.

    Returns
    -------
    list[int]
        The sorted list of integers.

    Notes
    -----
    Bubble sort has a time complexity of O(n^2) in the worst case, which is less efficient than other sorting algorithms like insertion sort and selection sort.
    However, bubble sort has the advantage of being simple to implement and being relatively efficient when the input is already partially sorted.
    """
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:  
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # Swap
                swapped = True
        if not swapped:  # Optimization: Stop if no swaps occurred
            break
    return arr

def selection_sort(arr: list[int]) -> list[int]:
    """
    Sorts the given list of integers in ascending order using the selection sort algorithm.

    The selection sort algorithm works by repeatedly selecting the minimum element from the unsorted subarray and swapping it with the first element of the unsorted subarray.
    The algorithm stops when all elements have been sorted.

    Parameters
    ----------
    arr : list[int]
        The list of integers to be sorted.

    Returns
    -------
    list[int]
        The sorted list of integers.

    Notes
    -----
    Selection sort has a time complexity of O(n^2) in the worst case, which is less efficient than other sorting algorithms like insertion sort and bubble sort.
    However, selection sort has the advantage of being simple to implement and being relatively efficient when the input is already partially sorted.
    """

    n = len(arr)
    for i in range(n):
        min_index = i  
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:  
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

def insertion_sort(arr: list[int]) -> list[int]:
    """
    Sorts the given list of integers in ascending order using the insertion sort algorithm.

    The insertion sort algorithm works by iterating through the list and inserting each element into its correct position in the sorted subarray.
    The algorithm has a time complexity of O(n^2) in the worst case.
    However, insertion sort has the advantage of being simple to implement and being relatively efficient when the input is already partially sorted.

    Parameters
    ----------
    arr : list[int]
        The list of integers to be sorted.

    Returns
    -------
    list[int]
        The sorted list of integers.

    Notes
    -----
    Insertion sort has a time complexity of O(n^2) in the worst case, which is less efficient than other sorting algorithms like merge sort and quick sort.
    However, insertion sort has the advantage of being simple to implement and being relatively efficient when the input is already partially sorted.
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:  
            arr[j + 1] = arr[j]  
            j -= 1
        arr[j + 1] = key  
    return arr


def sanitize_text(text: str, more_keywords: list[str] = []) -> tuple[str, str]:
    """
    Sanitizes the given text by blocking any strings that contain potentially
    dangerous keywords. The function takes an optional list of additional keywords
    to block. The function returns a tuple containing the sanitized text and an
    optional log message if any potentially dangerous keywords were found.

    Parameters
    ----------
    text : str
        The string to be sanitized.
    more_keywords : list[str], optional
        A list of additional keywords to block.

    Returns
    -------
    tuple[str, str]
        A tuple containing the sanitized text and an optional log message if any
        potentially dangerous keywords were found.

    Raises
    ------
    TypeError
        If more_keywords is not a list or if any of its elements are not strings.
    """
    import re
    log_message = ""

    if not isinstance(more_keywords, list) or not all(isinstance(k, str) for k in more_keywords):
        raise TypeError("more_keywords must be a list of strings")

    BLACKLISTED_KEYWORDS = [
        r"\bimport\b", r"\bexec\b", r"\beval\b", r"\bsystem\(", r"\bos\.", 
        r"\bsubprocess\.", r"\brm\s+-rf\b", r"\brmdir\b", r"\bdel\b", 
        r"\bopen\(", r"\bwrite\(", r"\bread\(", r"\bchmod\b", r"\bchown\b",  
    ]

    # Escape user-defined keywords to prevent regex injection
    escaped_keywords = [re.escape(k) for k in more_keywords]

    # Merge all keywords into a single regex pattern
    combined_pattern = "|".join(BLACKLISTED_KEYWORDS + escaped_keywords)

    # Find matches
    match = re.search(combined_pattern, text, re.IGNORECASE)
    
    if match:
        log_message = f"Blocked potentially dangerous input: '{text}' matches keyword '{match.group()}'"
        print_from("LogFileMonitor - Sanitize Text", log_message)
        return "", log_message

    return text, ""




def enumerate_directory(path: str, levels: int = 0) -> list[str | dict[str, list]]:
    import os
    def _scan_directory(directory: str, depth: int) -> list[str | dict[str, list]]:
        contents = []
        
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file():
                        contents.append(entry.name)
                    elif entry.is_dir():
                        if depth > 0:
                            subdir_contents = _scan_directory(entry.path, depth - 1)
                            contents.append({entry.name: subdir_contents})
                        else:
                            contents.append(entry.name)
        except PermissionError:
            print(f"Permission denied: {directory}")

        return contents

    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory does not exist: {path}")

    if not os.path.isdir(path):
        raise NotADirectoryError(f"Path is not a directory: {path}")

    return _scan_directory(path, levels)


def clipboard_get() -> str:
    """Retrieves the current contents of the system clipboard as a string.

    Returns
    -------
    str
        The contents of the clipboard.
    """
    import pyperclip
    return pyperclip.paste()

def clipboard_set(text: str) -> None:

    """Sets the contents of the system clipboard to the given text.

    Parameters
    ----------
    text : str
        The text to be copied to the clipboard.
    """
    import pyperclip
    pyperclip.copy(text)


def timed_delay(wait_time: float, variant_time_x: float = 0, variant_time_y: float = 0) -> None:
    """
    Blocks the current thread for a specified amount of time, with optional variability.

    Parameters
    ----------
    wait_time : float
        The base time in seconds to wait.
    variant_time_x : float, optional
        The minimum additional random time to add to the wait_time, defaults to 0.
    variant_time_y : float, optional
        The maximum additional random time to add to the wait_time, defaults to 0.

    Returns
    -------
    None
    """

    import time, random

    wait_time = max(0, wait_time)
    variant_time_x, variant_time_y = sorted([max(0, variant_time_x), max(0, variant_time_y)])

    total_delay = wait_time + random.uniform(variant_time_x, variant_time_y)
    print_from("OPR - time_delay", f"Waiting for {total_delay:.2f} seconds...")
    time.sleep(total_delay)


def random_within_boundary_box(x: float, y: float, h: float, w: float, centered: bool = False) -> tuple[int, int]:
    """
    Generates a random integer coordinate within a specified boundary box.

    If `centered` is True, the (x, y) coordinates represent the center of the boundary box,
    and the function adjusts the min/max values accordingly.

    Parameters
    ----------
    x : float
        The x-coordinate of the top-left corner (or center if centered=True) of the boundary box.
    y : float
        The y-coordinate of the top-left corner (or center if centered=True) of the boundary box.
    h : float
        The height of the boundary box.
    w : float
        The width of the boundary box.
    centered : bool, optional
        If True, the given (x, y) is treated as the center of the box instead of the top-left corner.

    Returns
    -------
    tuple[int, int]
        A tuple containing random x and y coordinates within the boundary box.

    Raises
    ------
    ValueError
        If the calculated min coordinates are greater than the max coordinates due to
        non-positive dimensions, indicating an invalid boundary box.
    """

    import random

    if centered:
        x_min, x_max = int(x - w / 2), int(x + w / 2)
        y_min, y_max = int(y - h / 2), int(y + h / 2)
    else:
        x_min, x_max = int(x), int(x + w)
        y_min, y_max = int(y), int(y + h)

    if x_min > x_max or y_min > y_max:
        raise ValueError(f"Invalid boundary box: ({x}, {y}, {h}, {w}). Ensure width and height are positive.")

    return random.randint(x_min, x_max), random.randint(y_min, y_max)


def file_move(source: str, destination: str) -> tuple[bool, str]:
    """
    Moves a file from one location to another.

    Parameters
    ----------
    source : str
        The source path of the file to be moved.
    destination : str
        The destination path where the file should be moved.

    Returns
    -------
    tuple[bool, str]
        A tuple where the first value indicates success (True/False), 
        and the second value contains an error message if failed.
    """
    import shutil, os

    # Ensure the source file exists
    if not os.path.exists(source):
        return False, f"Source file not found: {source}"

    # Ensure destination directory exists
    destination_dir = os.path.dirname(destination)
    if destination_dir and not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    try:
        shutil.move(source, destination)
        return True, "File moved successfully"
    except Exception as e:
        return False, f"Failed to move file: {e}"

    

def file_copy(source: str, destination: str) -> tuple[bool, str]:
    """
    Copies a file from one location to another.

    Parameters
    ----------
    source : str
        The source path of the file to be copied.
    destination : str
        The destination path where the file should be copied.

    Returns
    -------
    tuple[bool, str]
        A tuple where the first value indicates success (True/False),
        and the second value contains an error message if failed.
    """
    import shutil, os

    # Ensure the source file exists
    if not os.path.exists(source):
        return False, f"Source file not found: {source}"

    # Ensure destination directory exists
    destination_dir = os.path.dirname(destination)
    if destination_dir and not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    try:
        shutil.copy2(source, destination)  # copy2 preserves metadata
        return True, "File copied successfully"
    except Exception as e:
        return False, f"Failed to copy file: {e}"


def get_processes() -> list[str]:
    """
    Returns a list of currently running process names.

    Returns
    -------
    list[str]
        A list of process names (lowercased).
    """
    import psutil  # Lazy import

    return [p.info['name'].lower() for p in psutil.process_iter(['name']) if p.info['name']]


def kill_process(process_name: str, process_list: list[str] = None) -> bool:
    """
    Finds and kills the closest matching process.

    Parameters
    ----------
    process_name : str
        The name of the process to kill.
    process_list : list[str], optional
        A pre-fetched list of running process names for optimization.

    Returns
    -------
    bool
        True if a process was successfully killed, False otherwise.
    """
    import psutil

    # Fetch running processes if not provided
    process_list = process_list or get_processes()

    if not process_list:
        print("No running processes found.")
        return False

    best_match = find_best_match(process_name.lower(), process_list)

    if not best_match:
        print(f"No suitable match found for process: {process_name}")
        return False

    success = False
    for process in psutil.process_iter(['name']):
        if process.info['name'].lower() == best_match:
            try:
                process.kill()
                print(f"Killed process: {best_match}")
                success = True
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"Could not terminate {best_match}: {e}")

    return success



def get_main_idea(passage: str, sentences: int = 1, summarizer: str = 'lsa') -> str:
    """
    Gets the main idea of a given passage of text.

    Parameters
    ----------
    passage : str
        The text passage to summarize.
    sentences : int, optional
        The number of sentences to return in the summary. Defaults to 1.
    summarizer : str, optional
        The summarizer to use. Choose from 'lsa', 'lex_rank', 'luhn', or 'text_rank'. Defaults to 'lsa'.

    Returns
    -------
    str
        The main idea of the passage, summarized in the number of sentences specified.

    Raises
    ------
    ValueError
        If the specified summarizer is not valid.
    """
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.luhn import LuhnSummarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    import nltk
    nltk.download('punkt_tab')


    summarizers = {
        'luhn': LuhnSummarizer,
        'lex_rank': LexRankSummarizer,
        'lsa': LsaSummarizer,
        'text_rank': TextRankSummarizer,
    }

    if summarizer not in summarizers:
        raise ValueError(f"Invalid summarizer '{summarizer}'. Choose from: {', '.join(summarizers.keys())}")
    
    parser = PlaintextParser.from_string(passage, Tokenizer("english"))
    summarizer_instance = summarizers[summarizer]()
    summary = summarizer_instance(parser.document, sentences)
    
    return " ".join(str(sentence) for sentence in summary)


def print_from(name: str, message: str, return_count: int = 0) -> str:

    """
    Prints a formatted message with an optional number of preceding newlines, and returns the formatted string.

    Parameters
    ----------
    name : str
        The name to be used as a prefix in the message.
    message : str
        The message content to be printed.
    return_count : int, optional
        The number of newline characters to prepend before the message. Defaults to 0.

    Returns
    -------
    str
        The formatted string of the form "[{name}] {message}".
    """

    if return_count > 0:
        _ = "\n" * return_count
        print(_)

    _ = f"[{name}] {message}"

    print(_)

    return _

def input_from(name: str, message: str) -> str:

    """
    Asks the user for input with a name prefix and returns the input string.

    Parameters
    ----------
    name : str
        The name to be displayed as a prefix in the message.
    message : str
        The message content to be displayed before the input prompt.

    Returns
    -------
    str
        The string input by the user.
    """
    _ = input(f"[{name}] {message}: ")

    return _

def print_pretty(message: str, flourish: str, num: int) -> str:

    """
    Prints a formatted message with a flourish prefix and suffix, and returns the formatted string.

    Parameters
    ----------
    message : str
        The message content to be printed and returned.
    flourish : str
        The flourish string to be repeated at the start and end of the message.
    num : int
        The number of times the flourish should be repeated.

    Returns
    -------
    str
        The formatted string of the form "{flourish * num} message {flourish * num}".
    """

    _ = f"{flourish * num} {message} {flourish * num}"

    print(_)

    return _

def stubborn_call(*args, func: callable, stubborn: bool = False, wait_time: float = .5, whitelist: list[Exception] = [], whitelist_message: str = None, **kwargs) -> any:
    
    """
    Calls a function with the given arguments and keyword arguments, and if the function call fails (raises an exception),
    it will retry the function call after a specified wait time. If a whitelist of exceptions is provided, it will not
    re-raise the error if the exception is in the whitelist, and will print a message indicating that the error was expected.

    NOTE: Put non-keyword arguments as the starting arguments, before the callable.

    Parameters
    ----------
    *args : any
        Positional arguments to be passed to the callable.
    func : callable
        The function to be called.
    stubborn : bool, optional
        If set to True, the function will keep retrying until stopped, by default False.
    wait_time : float, optional
        The time to wait in seconds before retrying the function call, by default 0.5.
    whitelist : list[Exception], optional
        A list of exceptions to be considered expected and not re-raised, by default empty.
    whitelist_message : str, optional
        A custom message to be printed when an expected error is encountered, by default None.
    **kwargs : any
        Keyword arguments to be passed to the callable.

    Returns
    -------
    any
        The return value of the callable.
    """

    import time

    print_from("OPR - stubborn_call", f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")

    while True:
        
        try:

            return func(*args, **kwargs)    
        
        except Exception as e:    

            if any(isinstance(e, exc) for exc in whitelist):
                msg = whitelist_message if whitelist_message else f"Expected Error, retrying in {wait_time} seconds: {type(e).__name__}"
                print_from("OPR - stubborn_call", msg)
            
            else:
                print_from("OPR - stubborn_call", f"Unexpected Error: {e}")

                if not stubborn: 
                    raise e
                
            
            time.sleep(wait_time)
            continue
            


def hammer_call(*args, func: callable, stop_at: callable = None, stop_if: any = None, try_count: int = 3,  stubborn: bool = False, wait_time: float = .5, whitelist: list[Exception] = [], whitelist_message: str = None, **kwargs) -> any:

    """
    Calls a function multiple times until it the tries are exhausted, or a specified stop condition is met.

    NOTE: Put non-keyword arguments as the starting arguments, before the callable.

    Parameters
    ----------
    *args : any
        Positional arguments to be passed to the callable.
    func : callable
        The callable to be called.
    stop_at : callable, optional
        A function that takes one argument to check if the result of the callable meets a certain condition, by default None.
    stop_if : any, optional
        A value to check if the result of the callable is equal to, by default None.
    try_count : int, optional
        The number of times to call the callable, by default 3.
    stubborn : bool, optional
        Whether to keep trying even if an unexpected error occurs, by default False.
    wait_time : float, optional
        The time in seconds to wait between calls, by default 0.5.
    whitelist : list[Exception], optional
        A list of exceptions that are expected to occur and should be ignored, by default [].
    whitelist_message : str, optional
        A custom message to be printed when an expected error is encountered, by default None.
    **kwargs : any
        Keyword arguments to be passed to the callable.

    Returns
    -------
    any
        The return value of the callable.
    """
    import time

    print_from("OPR - hammer_call", f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")

    result = None
        
    for _ in range(try_count):
            
        try:

            result = func(*args, **kwargs)

            if stop_at is not None and stop_at(result) == stop_if:
                return result

        except Exception as e:

            if any(isinstance(e, exc) for exc in whitelist):
                msg = whitelist_message if whitelist_message else f"Expected Error, retrying in {wait_time} seconds: {type(e).__name__}"

                print_from("OPR - hammer_call", msg)

            else:
                print_from("OPR - hammer_call", f"Unexpected Error: {e}")

                if not stubborn: 
                    raise e           
            
        
        time.sleep(wait_time)

    return result

    
def clean_path(path: str) -> str:
    """
    Cleans the given file path by removing leading command symbols, 
    extraneous surrounding quotes, and redundant quote characters. 
    Normalizes the path for the operating system.

    NOTE: This is to be used for file paths that are dragged and dropped onto the terminal.

    Parameters
    ----------
    path : str
        The file path to be cleaned.

    Returns
    -------
    str
        The cleaned and normalized file path.
    """

    import os
    
    if path.startswith("& "):
        path = path[2:]

    if (path.startswith("'") and path.endswith("'")) or (path.startswith('"') and path.endswith('"')):
        path = path[1:-1]

    if "''" in path:
        path = path.replace("''", "'")

    return os.path.normpath(path)


def load_json(is_from: str, path: str, filename: str = "config.json"):

    """
    Loads a JSON file from a given path and filename.

    Parameters
    ----------
    is_from : str
        The name of the module or function calling this function, used for printing.
    path : str
        The path to the directory containing the JSON file.
    filename : str, optional
        The filename of the JSON file to be loaded, by default "config.json".

    Returns
    -------
    dict
        The loaded JSON file as a dictionary.
    """
    import json, os

    print_from(is_from, "Loading config file")
    config_file_path = os.path.join(path, filename)
    
    if not os.path.exists(config_file_path):
        with open(config_file_path, "w") as f:
            print_from(is_from, f"{filename} not found, creating empty file")
            f.write("{}")
    
    with open(config_file_path, "r") as f:
        print_from(is_from, f"SUCCESS: Loaded {filename}")
        return json.load(f)

def save_json(is_from: str, path: str, dump: dict, filename: str = "config.json", indent: int = 4):

    import json, os

    print_from(is_from, "Saving config file")

    config_file_path = os.path.join(path, filename)
    
    os.makedirs(path, exist_ok=True)

    with open(config_file_path, "w") as f:
        json.dump(dump, f, indent = indent)
        print_from(is_from, f"SUCCESS: Saved {filename}")