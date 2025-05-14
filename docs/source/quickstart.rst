Quickstart
==========

This guide will help you get started with Feluda quickly.

Basic Usage
----------

Feluda can be used as a Python library to process and analyze data using its modular operators.

Configuration
~~~~~~~~~~~~

Feluda uses a configuration file (``.yml``) to define the operators and their parameters. This allows you to customize your workflow without modifying the code.

Here's an example configuration file (``config.yml``):

.. code-block:: yaml

    operators:
      label: "Operators"
      parameters:
        - name: "Video Vector Representation"
          type: "vid_vec_rep_clip"
          parameters: {}
        - name: "Image Vector Representation"
          type: "image_vec_rep_resnet"
          parameters: {}

- **``operators``**: A list of operators to be used.
- **``name``**: The name of the operator.
- **``parameters``**: Any other operator-specific parameters.

Code Example
~~~~~~~~~~~

Here's a simple example to demonstrate how to use Feluda:

.. code-block:: python

    from feluda import Feluda

    config_path = "/path/to/config.yml"

    # Initialize Feluda with the configuration file
    feluda = Feluda(config_path)
    
    # Set up Feluda and its operators
    feluda.setup()

    # Access an operator and run a task
    operator = feluda.operators.get()["vid_vec_rep_clip"]
    result = operator.run("path/to/example.mp4")
    print(result)

Example Use Cases
----------------

Video Embedding and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract video embeddings and visualize them in 2D space using t-SNE:

.. code-block:: python

    from feluda import Feluda
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Initialize Feluda
    config_path = "config.yml"
    feluda = Feluda(config_path)
    feluda.setup()
    
    # Get operators
    vid_vec_rep = feluda.operators.get()["vid_vec_rep_clip"]
    dimension_reduction = feluda.operators.get()["dimension_reduction"]
    
    # Process videos
    video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
    embeddings = []
    
    for video_path in video_paths:
        embedding = vid_vec_rep.run(video_path)
        embeddings.append(embedding)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Reduce dimensions for visualization
    reduced_embeddings = dimension_reduction.run(embeddings_array)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    plt.title("Video Embeddings Visualization")
    plt.show()

Video Clustering
~~~~~~~~~~~~~~

Automatically cluster videos based on their content:

.. code-block:: python

    from feluda import Feluda
    import numpy as np
    
    # Initialize Feluda
    config_path = "config.yml"
    feluda = Feluda(config_path)
    feluda.setup()
    
    # Get operators
    vid_vec_rep = feluda.operators.get()["vid_vec_rep_clip"]
    cluster_embeddings = feluda.operators.get()["cluster_embeddings"]
    
    # Process videos
    video_paths = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
    embeddings = []
    
    for video_path in video_paths:
        embedding = vid_vec_rep.run(video_path)
        embeddings.append(embedding)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Cluster videos
    clusters = cluster_embeddings.run(embeddings_array)
    
    # Print clusters
    for cluster_name, cluster_videos in clusters.items():
        print(f"Cluster: {cluster_name}")
        for video_idx in cluster_videos:
            print(f"  - {video_paths[video_idx]}")

Next Steps
---------

- Check out the :doc:`operators` page for a list of available operators
- Learn about the :doc:`architecture` of Feluda
- See the :doc:`api` reference for detailed documentation
