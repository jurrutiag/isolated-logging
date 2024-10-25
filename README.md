# Isolated Logging

**Isolated Logging** is a lightweight Python library for tracking and logging the performance of functions and loops showing them in an isolated way (color-coding and dedicated loggers or files). It helps developers assess and optimize their code by monitoring execution times and calculating statistics like average time per iteration, standard deviations, and estimated completion times.

## Features

- **Function and Loop Execution Time Tracking:** Measures execution times and logs details about each iteration.
- **Performance Statistics:** Provides averages, standard deviations, and estimated time remaining.
- **Logging to Temporary or Specified File:** Option to log messages and statistics to a temporary log file or a custom file path.
- **Color-coded Console Output:** Uses ANSI escape codes to improve readability in terminal output.
- **Customizable Logging Options:** Flexible setup options for independent logging, with decorator-based function timing and loop performance monitoring.

## Installation

Simply clone this repository:

```bash
git clone https://github.com/jurrutiag/isolated-logging.git
```

## Usage

### 1. Setting Up Logging

To begin, set up logging by calling `setup_log_file_and_logger()` at the start of your script. For this you either have to provide a logger or set `setup_independent_logging=True`.

```python
from isolated_logging import setup_log_file_and_logger

# Set up logging to a temporary file
setup_log_file_and_logger(setup_independent_logging=True)
```

### 2. Timing Functions

Use the `log_timed_function` decorator to time a function and track its execution stats.

```python
from isolated_logging import log_timed_function

@log_timed_function(ignore_instant_returns=True, threshold=0.001)
def your_function():
    # Your function logic
    pass
```

### 3. Timing Loops

Wrap your loops with `log_timed_loop` to measure and log each iteration’s execution time. If the iterable is a list it will also calculate ETA and tell you how many loops are left.

```python
from isolated_logging import log_timed_loop

for item in log_timed_loop(your_iterable, loop_name="Processing Items"):
    # Process each item
    pass
```

## Example

```python
import time
from isolated_logging import log_timed_function, log_timed_loop, setup_log_file_and_logger

setup_log_file_and_logger(setup_independent_logging=True)

@log_timed_function(ignore_instant_returns=True, threshold=0.001)
def sample_function(n):
    time.sleep(n)

for _ in log_timed_loop(range(5), loop_name="Sample Loop"):
    sample_function(0.5)
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.