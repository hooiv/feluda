Contributing
============

Thank you for your interest in contributing to Feluda! This document provides guidelines and instructions for contributing to the project.

Getting Started
--------------

1. Fork the repository on GitHub.
2. Clone your fork locally:

   .. code-block:: bash

       git clone https://github.com/your-username/feluda.git
       cd feluda

3. Set up a development environment:

   .. code-block:: bash

       # Using pip
       pip install -e ".[dev]"
       
       # Or using uv
       uv pip install -e ".[dev]"

4. Install pre-commit hooks:

   .. code-block:: bash

       pre-commit install

Development Workflow
------------------

1. Create a new branch for your feature or bugfix:

   .. code-block:: bash

       git checkout -b feature/your-feature-name

2. Make your changes, following the coding standards and guidelines.

3. Run the tests to ensure your changes don't break existing functionality:

   .. code-block:: bash

       pytest

4. Run the linters to ensure your code follows the project's style guidelines:

   .. code-block:: bash

       ruff check .
       black --check .
       mypy feluda

5. Commit your changes with a descriptive commit message:

   .. code-block:: bash

       git commit -m "feat: add your feature description"

6. Push your changes to your fork:

   .. code-block:: bash

       git push origin feature/your-feature-name

7. Create a pull request on GitHub.

Coding Standards
--------------

- Follow PEP 8 style guidelines.
- Use type hints for all function and method signatures.
- Write docstrings for all functions, methods, and classes using Google style.
- Use contract programming with the `deal` library for critical functions.
- Write tests for all new functionality.

Testing
------

- Write unit tests for all new functionality.
- Use pytest for running tests.
- Use hypothesis for property-based testing where appropriate.
- Aim for high test coverage (>90%).

Documentation
------------

- Write clear and concise documentation for all new features.
- Update existing documentation when changing functionality.
- Use Sphinx for generating documentation.
- Include examples and use cases where appropriate.

Formal Verification
-----------------

For critical components, we use formal verification to ensure correctness:

1. Use the `deal` library to specify contracts (preconditions, postconditions, and invariants).
2. Use CrossHair to verify these contracts statically.
3. Run the verification script to check your changes:

   .. code-block:: bash

       python scripts/verify_vector_operations.py

Pull Request Process
------------------

1. Ensure your code passes all tests and linters.
2. Update the documentation to reflect any changes.
3. Update the CHANGELOG.md file with a description of your changes.
4. The pull request will be reviewed by maintainers, who may request changes.
5. Once approved, your pull request will be merged.

Release Process
-------------

1. Update the version number in pyproject.toml.
2. Update the CHANGELOG.md file with a description of the changes.
3. Create a new tag with the version number:

   .. code-block:: bash

       git tag -a v1.0.0 -m "Release v1.0.0"
       git push origin v1.0.0

4. The release workflow will automatically build and publish the package to PyPI.

Code of Conduct
-------------

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms. See the CODE_OF_CONDUCT.md file for details.
