Operators
=========

Operators are the building blocks of Feluda. Each operator performs a specific task, such as extracting vector representations from images or videos, detecting text in images, clustering embeddings, or reducing dimensionality.

Available Operators
------------------

Image Vector Representation (ResNet)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `image_vec_rep_resnet` operator extracts vector representations (embeddings) from images using ResNet models.

.. code-block:: yaml

    operators:
      label: "Operators"
      parameters:
        - name: "Image Vector Representation"
          type: "image_vec_rep_resnet"
          parameters:
            model_name: "resnet50"  # Optional, default: "resnet50"
            use_pretrained: true    # Optional, default: true
            device: "auto"          # Optional, default: "auto"
            batch_size: 32          # Optional, default: 32
            normalize_embeddings: true  # Optional, default: true

Video Vector Representation (CLIP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `vid_vec_rep_clip` operator extracts vector representations (embeddings) from videos using CLIP models.

.. code-block:: yaml

    operators:
      label: "Operators"
      parameters:
        - name: "Video Vector Representation"
          type: "vid_vec_rep_clip"
          parameters: {}

Dimension Reduction
~~~~~~~~~~~~~~~~~

The `dimension_reduction` operator reduces the dimensionality of embeddings using techniques like t-SNE or UMAP.

.. code-block:: yaml

    operators:
      label: "Operators"
      parameters:
        - name: "Dimension Reduction"
          type: "dimension_reduction"
          parameters:
            model_type: "tsne"  # Optional, default: "tsne"
            n_components: 2     # Optional, default: 2
            perplexity: 30      # Optional, default: 30 (for t-SNE)
            n_neighbors: 15     # Optional, default: 15 (for UMAP)

Cluster Embeddings
~~~~~~~~~~~~~~~~

The `cluster_embeddings` operator clusters embeddings using techniques like K-means or Agglomerative Clustering.

.. code-block:: yaml

    operators:
      label: "Operators"
      parameters:
        - name: "Cluster Embeddings"
          type: "cluster_embeddings"
          parameters:
            model_type: "kmeans"  # Optional, default: "kmeans"
            n_clusters: 5         # Optional, default: 5

Detect Text in Image (Tesseract)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `detect_text_in_image_tesseract` operator extracts text from images using Tesseract OCR.

.. code-block:: yaml

    operators:
      label: "Operators"
      parameters:
        - name: "Detect Text in Image"
          type: "detect_text_in_image_tesseract"
          parameters:
            psm: 6  # Optional, default: 6
            oem: 1  # Optional, default: 1

Detect Lewd Images
~~~~~~~~~~~~~~~~

The `detect_lewd_images` operator detects inappropriate content in images.

.. code-block:: yaml

    operators:
      label: "Operators"
      parameters:
        - name: "Detect Lewd Images"
          type: "detect_lewd_images"
          parameters: {}

Video Hash TMK
~~~~~~~~~~~~

The `video_hash_tmk` operator generates perceptual hashes for videos using TMK+PDQF.

.. code-block:: yaml

    operators:
      label: "Operators"
      parameters:
        - name: "Video Hash TMK"
          type: "video_hash_tmk"
          parameters: {}

Classify Video Zero Shot
~~~~~~~~~~~~~~~~~~~~~

The `classify_video_zero_shot` operator classifies videos using zero-shot learning.

.. code-block:: yaml

    operators:
      label: "Operators"
      parameters:
        - name: "Classify Video Zero Shot"
          type: "classify_video_zero_shot"
          parameters:
            classes: ["cat", "dog", "bird"]  # Required

Creating Custom Operators
-----------------------

You can create custom operators by implementing the operator interface. For basic operators, you need to create a module with `initialize` and `run` functions:

.. code-block:: python

    def initialize(parameters):
        """Initialize the operator."""
        global some_global_variable
        some_global_variable = parameters.get("some_parameter", default_value)

    def run(input_data):
        """Run the operator."""
        # Process the input data
        return result

For more advanced operators, you can inherit from the `BaseFeludaOperator` class, which provides a standardized interface and contract enforcement:

.. code-block:: python

    from feluda.base_operator import BaseFeludaOperator
    from pydantic import BaseModel, Field

    class MyOperatorParameters(BaseModel):
        """Parameters for my operator."""
        param1: str = Field(default="default_value")
        param2: int = Field(default=42)

    class MyOperator(BaseFeludaOperator[InputType, OutputType, MyOperatorParameters]):
        """My custom operator."""
        
        name = "MyOperator"
        description = "A custom operator"
        version = "1.0.0"
        parameters_model = MyOperatorParameters
        
        def _initialize(self) -> None:
            """Initialize the operator."""
            # Initialization code here
            pass
        
        def _execute(self, input_data: InputType) -> OutputType:
            """Execute the operator."""
            # Processing code here
            return result

    # For backward compatibility
    def initialize(parameters):
        """Initialize the operator."""
        global operator
        operator = MyOperator(parameters=parameters)

    def run(input_data):
        """Run the operator."""
        global operator
        return operator._execute(input_data)
