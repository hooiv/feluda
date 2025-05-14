Installation
============

Prerequisites
------------

Before you begin, ensure that the following system dependencies are installed:

* Python version 3.10 or higher
* Optionally we recommend using ``uv`` for Python package and project management. Install ``uv`` by following its `official installation guide <https://docs.astral.sh/uv/>`_.

Installing Feluda
----------------

You can install ``feluda`` using pip:

.. code-block:: bash

    pip install feluda

Each operator also has to be installed separately. For instance, you can install the ``feluda-vid-vec-rep-clip`` operator like:

.. code-block:: bash

    pip install feluda-vid-vec-rep-clip

For a list of published Feluda operators, see the `Tattle PyPI page <https://pypi.org/user/tattle/>`_.

Development Installation
----------------------

For development, you can install Feluda with all development dependencies:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/tattle-made/feluda.git
    cd feluda
    
    # Install with development dependencies
    pip install -e ".[dev]"
    
    # Or using uv
    uv pip install -e ".[dev]"

Additional Dependencies
---------------------

Feluda has several optional dependency groups that you can install based on your needs:

Security Tools:

.. code-block:: bash

    pip install -e ".[security]"

Performance Tools:

.. code-block:: bash

    pip install -e ".[performance]"

Machine Learning Tools:

.. code-block:: bash

    pip install -e ".[ml]"

You can also install multiple dependency groups:

.. code-block:: bash

    pip install -e ".[dev,security,performance]"

Using the Development Container
-----------------------------

For the easiest development setup, we provide a devcontainer configuration that works with Visual Studio Code:

1. Install `Visual Studio Code <https://code.visualstudio.com/>`_
2. Install the `Remote - Containers <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`_ extension
3. Clone the Feluda repository
4. Open the repository in VS Code
5. When prompted, click "Reopen in Container"

The devcontainer includes all necessary dependencies and tools for development.
