import atexit
import os
import tempfile
import time
from collections import defaultdict
from functools import wraps

import numpy as np

# Global variable to hold the reference to the log file
_log_file = None
_logger = None

# Dictionary to track function statistics
_function_stats = defaultdict(list)  # Key: function name, Value: list of execution times

# Dictionary to track loop iteration statistics
_loop_stats = defaultdict(
    lambda: {"times": [], "iterations": 0}
)  # Key: loop identifier, Value: dict with times and count

_checkpoint_stats = defaultdict(lambda: {"times": [], "last_checkpoint": None})

# ANSI escape codes for colors
COLOR_RESET = "\033[0m"
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_CYAN = "\033[96m"
COLOR_GRAY = "\033[90m"
COLOR_PURPLE = "\033[95m"
COLOR_DULL = "\033[37m"  # Dull white/gray color
COLOR_ORANGE = "\033[38;5;214m"


def setup_log_file_and_logger(path=None, logger=None, setup_independent_logging=False):
    """Sets up the log file if it hasn't been set up already and prints its location."""
    global _log_file
    global _logger
    if setup_independent_logging:
        if _log_file is None:
            if path is None:
                _log_file = tempfile.NamedTemporaryFile(delete=False, mode="w+", suffix=".log")

            else:
                _log_file = open(path, "w+")

            print(f"{COLOR_CYAN}Log file created at: {_log_file.name}{COLOR_RESET}")
            atexit.register(close_log)  # Register the close_log function to be called at exit

    if _logger is None:
        _logger = logger

    if _log_file is None and _logger is None:
        print(
            f"{COLOR_RED}No log file or logger set up!, run setup_log_file_and_logger first with setup_independent_logging or logger arguments.{COLOR_RESET}"
        )

    return _log_file, _logger


def _log_raw_message(message, end="\n"):
    log_file, logger = setup_log_file_and_logger()
    if log_file is not None:
        log_file.write(f"{message}{end}")
        log_file.flush()

    if logger is not None:
        logger.info(message)


def log_message(message):
    """Logs a simple message to the temp log file."""
    colored_message = f"{COLOR_BLUE}{message}{COLOR_RESET}"
    _log_raw_message(f"{colored_message}")


def log_list_elements(elements):
    """Logs all elements in a list to the temp log file."""
    _log_raw_message(f"{COLOR_GREEN}Logging list elements:{COLOR_RESET}")
    for element in elements:
        _log_raw_message(f"{COLOR_BLUE}- {element}{COLOR_RESET}")


def log_timed_loop(
    iterable, ignore_instant_iterations=True, threshold=0.001, loop_id=None, loop_name=None
):
    """Logs the time taken for each iteration in a loop, calculates ETA if possible, and yields each element."""

    total_items = len(iterable) if hasattr(iterable, "__len__") else None
    start_time = time.time()
    loop_identifier = loop_id or id(iterable)  # Unique identifier for the loop

    instant_items = 0

    loop_name_str = f"({loop_name}) " if loop_name else ""

    for i, item in enumerate(iterable):
        iter_start_time = time.time()
        _log_raw_message(
            f"{COLOR_BLUE}{loop_name_str}Iteration {i+1}: Started processing {COLOR_GRAY}{item}{COLOR_RESET}"
        )

        yield item  # Yield the current item for the loop to use

        iter_end_time = time.time()
        iter_elapsed_time = iter_end_time - iter_start_time
        _log_raw_message(
            f"{COLOR_GREEN}{loop_name_str}Iteration {i+1}: Finished processing {COLOR_GRAY}{item} {COLOR_YELLOW}in {iter_elapsed_time:.2f} seconds{COLOR_RESET}"
        )

        if ignore_instant_iterations and iter_elapsed_time < threshold:
            instant_items += 1
            continue

        # Store iteration time
        _loop_stats[loop_identifier]["times"].append(iter_elapsed_time)
        _loop_stats[loop_identifier]["iterations"] += 1

        # Calculate and log ETA if the total length is known
        if total_items is not None:
            completed_items = i + 1 - (instant_items if ignore_instant_iterations else 0)
            total_elapsed_time = iter_end_time - start_time
            average_time_per_item = total_elapsed_time / completed_items
            remaining_items = total_items - completed_items - instant_items
            eta = remaining_items * average_time_per_item
            _log_raw_message(
                f"{COLOR_YELLOW}{loop_name_str}Iteration {i+1}: Estimated time remaining: {COLOR_YELLOW}{eta:.2f} seconds ({remaining_items} iterations remaining, Average time per item: {average_time_per_item:.2f} [s]){COLOR_RESET}"
            )

    # Log loop statistics
    average_iter_time = np.mean(_loop_stats[loop_identifier]["times"])
    iterations_per_sec = 1 / average_iter_time if average_iter_time > 0 else float("inf")
    _log_raw_message(
        f"{COLOR_DULL}{loop_name_str}Loop ID {loop_identifier}: Average time per iteration: {average_iter_time:.2f} seconds ({iterations_per_sec:.2f} iterations per second). Total iterations: {_loop_stats[loop_identifier]['iterations']}.{COLOR_RESET}"
    )


def log_timed_function(ignore_instant_returns, threshold=0.001):
    """Decorator factory that logs the timing of a function call and tracks statistics.
    - ignore_instant_returns: If True, ignores instant returns for counting time.
    - threshold: The time threshold below which the return is considered instant (default 1 millisecond).
    """

    def decorator(func):
        @wraps(func)  # This line ensures the original function's metadata is preserved
        def wrapper(*args, **kwargs):
            _log_raw_message(
                f"{COLOR_BLUE}Function {func.__name__} started with args: {COLOR_GRAY}{args}{COLOR_CYAN} and kwargs: {COLOR_GRAY}{kwargs}{COLOR_RESET}"
            )

            start_time = time.time()
            result = func(*args, **kwargs)  # Run the actual function with its arguments
            elapsed_time = time.time() - start_time

            # If ignoring instant returns and the time is below the threshold, skip logging
            if ignore_instant_returns and elapsed_time < threshold:
                return result

            # Store function execution time
            _function_stats[func.__name__].append(elapsed_time)

            _log_raw_message(
                f"{COLOR_GREEN}Function {func.__name__} finished {COLOR_YELLOW}in {elapsed_time:.2f} seconds{COLOR_RESET}"
            )

            # Calculate statistics
            avg_time = np.mean(_function_stats[func.__name__])
            std_dev_time = (
                np.std(_function_stats[func.__name__])
                if len(_function_stats[func.__name__]) > 1
                else 0.0
            )
            num_calls = len(_function_stats[func.__name__])
            _log_raw_message(
                f"{COLOR_DULL}Function {func.__name__}: Avg time: {avg_time:.2f} seconds, Std dev: {std_dev_time:.2f}, Calls: {num_calls}{COLOR_RESET}"
            )

            return result

        return wrapper

    return decorator


def log_checkpoint(checkpoint_id, message_on_reach=None):
    """
    Logs a checkpoint with a unique identifier, tracks the time since the last checkpoint,
    and records relevant statistics for analysis.

    Parameters:
    - checkpoint_id (str): Unique identifier for the checkpoint.
    - message_on_reach (str, optional): Custom message to log each time the checkpoint is reached.
    """
    log_file, logger = setup_log_file_and_logger()
    current_time = time.time()

    # Log custom message if provided
    if message_on_reach:
        _log_raw_message(f"{COLOR_CYAN}Checkpoint {checkpoint_id}: {message_on_reach}{COLOR_RESET}")

    # Get the last checkpoint time if it exists
    last_time = _checkpoint_stats[checkpoint_id]["last_checkpoint"]
    if last_time is not None:
        # Calculate elapsed time since the last checkpoint
        elapsed_time = current_time - last_time
        _checkpoint_stats[checkpoint_id]["times"].append(elapsed_time)
        _log_raw_message(
            f"{COLOR_PURPLE}Checkpoint {checkpoint_id}: Elapsed time since last checkpoint: {elapsed_time:.2f} seconds{COLOR_RESET}"
        )
    else:
        _log_raw_message(
            f"{COLOR_PURPLE}Checkpoint {checkpoint_id}: Initial checkpoint recorded{COLOR_RESET}"
        )

    # Update the last checkpoint time to the current time
    _checkpoint_stats[checkpoint_id]["last_checkpoint"] = current_time

    # Calculate average time between checkpoints for this ID
    if _checkpoint_stats[checkpoint_id]["times"]:
        avg_time = np.mean(_checkpoint_stats[checkpoint_id]["times"])
        std_dev_time = (
            np.std(_checkpoint_stats[checkpoint_id]["times"])
            if len(_checkpoint_stats[checkpoint_id]["times"]) > 1
            else 0.0
        )
        num_checks = len(_checkpoint_stats[checkpoint_id]["times"])
        _log_raw_message(
            f"{COLOR_DULL}Checkpoint {checkpoint_id}: Avg time: {avg_time:.2f} seconds, Std dev: {std_dev_time:.2f}, Total checks: {num_checks}{COLOR_RESET}"
        )


def close_log():
    """Closes the log file if it's open."""
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None


def read_log():
    """Reads and returns the content of the log file."""
    log_file, logger = setup_log_file_and_logger()
    with open(log_file.name, "r") as file:
        return file.read()


def cleanup_log():
    """Cleans up (removes) the temp log file."""
    global _log_file
    if _log_file:
        os.remove(_log_file.name)
        _log_file = None
