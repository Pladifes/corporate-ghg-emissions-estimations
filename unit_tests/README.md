# Unit Test

This folder contains unit tests for our project, and we use pytest to run them.

## Requirements
The pytest package is already installed in the virtual environment. In case you don't use You can install it by running the following command:

    pip install pytest

## Running the Unit Tests
To run the unit tests, navigate to this folder in your terminal and run the following command:

    pytest

This will automatically discover and run all the tests in this folder.
to test a specific module, run the following command :

    pytest -k module_name

## Writing Unit Tests
If you need to write additional unit tests, please follow these guidelines:

- All test files should be named test_<module_name>.py.
- All test functions should be named test_<function_name>.
- Use assert statements to check the output of functions.
