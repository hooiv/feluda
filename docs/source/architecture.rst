Architecture
============

Feluda is designed with a modular architecture that allows for flexibility and extensibility. This document provides an overview of the architecture and the key components of the system.

Overview
--------

Feluda is built around the concept of operators, which are modular components that perform specific tasks. These operators can be combined to create complex workflows for analyzing multi-lingual and multi-modal content.

The core of Feluda is responsible for loading and configuring operators based on a configuration file, and for providing a unified interface to these operators.

Key Components
-------------

.. image:: _static/feluda_architecture.png
   :alt: Feluda Architecture
   :align: center

Core Components
~~~~~~~~~~~~~~

- **Feluda**: The main entry point for the library. It loads the configuration and sets up the operators.
- **Operator**: Manages the loading and initialization of operators.
- **BaseFeludaOperator**: The abstract base class that all operators must inherit from. It provides a standardized interface and contract enforcement.
- **Config**: Loads and validates the configuration from a YAML file.

Operators
~~~~~~~~

Operators are the building blocks of Feluda. Each operator performs a specific task, such as:

- Extracting vector representations from images or videos
- Detecting text in images
- Clustering embeddings
- Reducing dimensionality of embeddings
- Classifying videos

Operators can be combined to create complex workflows. For example, you might use an operator to extract vector representations from videos, another to reduce the dimensionality of these representations, and a third to cluster the videos based on these representations.

Formal Verification
~~~~~~~~~~~~~~~~~

Feluda includes components that are formally verified using tools like CrossHair and deal. These components are designed to be provably correct, with contracts that specify their behavior.

The `feluda.verification` package contains pure functions that are formally verified, ensuring that they behave correctly under all possible inputs.

Contract Programming
~~~~~~~~~~~~~~~~~~

Feluda uses contract programming to enforce preconditions, postconditions, and invariants on its components. This helps to catch errors early and ensures that components behave as expected.

The `deal` library is used to specify contracts on functions and methods, which are checked at runtime. These contracts can also be verified statically using tools like CrossHair.

Data Flow
--------

1. The user creates a configuration file that specifies the operators to use and their parameters.
2. The user initializes Feluda with the configuration file.
3. Feluda loads and initializes the operators based on the configuration.
4. The user can then access the operators through Feluda and use them to process data.

Example:

.. code-block:: python

    from feluda import Feluda

    # Initialize Feluda with a configuration file
    feluda = Feluda("config.yml")
    feluda.setup()

    # Access an operator and use it to process data
    operator = feluda.operators.get()["vid_vec_rep_clip"]
    result = operator.run("path/to/video.mp4")

Extensibility
-----------

Feluda is designed to be extensible. New operators can be added by creating a new module that implements the operator interface.

To create a new operator:

1. Create a new module with an `initialize` function that sets up the operator.
2. Implement a `run` function that processes the input data.
3. Add the operator to the configuration file.

For more advanced operators, you can inherit from the `BaseFeludaOperator` class, which provides a standardized interface and contract enforcement.
