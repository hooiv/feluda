Deployment Guide
===============

This guide provides instructions for deploying Feluda in various environments, from development to production.

Local Deployment
--------------

For local development and testing, you can run Feluda directly on your machine.

Running the CLI
~~~~~~~~~~~~~~

The Feluda CLI provides a command-line interface for running Feluda operations:

.. code-block:: bash

    # Run an operator
    feluda run --config config.yml --operator TextSentimentAnalysis --input input.json --output output.json

    # Verify a module
    feluda verify --module feluda.verification.vector_operations --verifier deal

    # Optimize parameters
    feluda optimize --config optimization.yml --output best_params.json

    # Run self-healing
    feluda heal --config healing.yml

    # Run an agent swarm
    feluda agent dev-swarm --task "Implement a function" --api-key "your-api-key"

    # Run a QA agent
    feluda agent qa --repo /path/to/repo --file /path/to/file.py --api-key "your-api-key"

Running the API
~~~~~~~~~~~~~

The Feluda API provides a web API for Feluda operations:

.. code-block:: bash

    # Run the API
    feluda-api --host 0.0.0.0 --port 8000 --config config.yml

You can then access the API at http://localhost:8000. The API documentation is available at http://localhost:8000/docs.

Running the Dashboard
~~~~~~~~~~~~~~~~~~

The Feluda Dashboard provides a web interface for monitoring and visualizing Feluda operations:

.. code-block:: bash

    # Run the dashboard
    feluda-dashboard --host 0.0.0.0 --port 8050

You can then access the dashboard at http://localhost:8050.

Docker Deployment
---------------

Feluda provides Docker images for easy deployment in containerized environments.

Using the Docker Image
~~~~~~~~~~~~~~~~~~~

You can run Feluda using the provided Docker image:

.. code-block:: bash

    # Pull the Docker image
    docker pull ghcr.io/tattle-made/feluda:latest

    # Run the API
    docker run -p 8000:8000 -v /path/to/config:/app/config ghcr.io/tattle-made/feluda:latest python -m feluda.api --config /app/config/config.yml

    # Run the dashboard
    docker run -p 8050:8050 ghcr.io/tattle-made/feluda:latest python -m feluda.dashboard

Using Docker Compose
~~~~~~~~~~~~~~~~~

Feluda provides Docker Compose configurations for running Feluda with its dependencies:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/tattle-made/feluda.git
    cd feluda

    # Run with Docker Compose
    docker-compose up

This will start the Feluda API, along with Elasticsearch and RabbitMQ.

For monitoring, you can use the monitoring stack:

.. code-block:: bash

    # Run the monitoring stack
    docker-compose -f docker-compose.monitoring.yml up

This will start Prometheus, Grafana, and Jaeger for monitoring and tracing.

Kubernetes Deployment
------------------

Feluda can be deployed on Kubernetes for production environments.

Using Helm
~~~~~~~~

Feluda provides Helm charts for deploying on Kubernetes:

.. code-block:: bash

    # Add the Feluda Helm repository
    helm repo add feluda https://tattle-made.github.io/feluda/charts
    helm repo update

    # Install Feluda
    helm install feluda feluda/feluda --values values.yaml

Example values.yaml:

.. code-block:: yaml

    api:
      enabled: true
      replicas: 2
      resources:
        requests:
          cpu: 100m
          memory: 256Mi
        limits:
          cpu: 500m
          memory: 512Mi

    dashboard:
      enabled: true
      replicas: 1
      resources:
        requests:
          cpu: 100m
          memory: 256Mi
        limits:
          cpu: 500m
          memory: 512Mi

    elasticsearch:
      enabled: true
      replicas: 3

    rabbitmq:
      enabled: true
      replicas: 3

    monitoring:
      enabled: true
      prometheus:
        enabled: true
      grafana:
        enabled: true
      jaeger:
        enabled: true

Using Kubernetes Manifests
~~~~~~~~~~~~~~~~~~~~~~~

You can also deploy Feluda using Kubernetes manifests:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/tattle-made/feluda.git
    cd feluda

    # Apply the manifests
    kubectl apply -f kubernetes/

Cloud Deployment
-------------

Feluda can be deployed on various cloud platforms.

AWS
~~~

To deploy Feluda on AWS, you can use the provided CloudFormation template:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/tattle-made/feluda.git
    cd feluda

    # Deploy using CloudFormation
    aws cloudformation create-stack --stack-name feluda --template-body file://aws/cloudformation.yml --parameters file://aws/parameters.json

Azure
~~~~~

To deploy Feluda on Azure, you can use the provided Azure Resource Manager template:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/tattle-made/feluda.git
    cd feluda

    # Deploy using Azure CLI
    az deployment group create --resource-group feluda --template-file azure/template.json --parameters azure/parameters.json

Google Cloud
~~~~~~~~~~

To deploy Feluda on Google Cloud, you can use the provided Deployment Manager template:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/tattle-made/feluda.git
    cd feluda

    # Deploy using gcloud
    gcloud deployment-manager deployments create feluda --config gcp/deployment.yaml

Production Considerations
----------------------

When deploying Feluda in production, consider the following:

Security
~~~~~~~

- Use HTTPS for all communications
- Set up authentication and authorization
- Use secrets management for sensitive information
- Regularly update dependencies
- Run security scans

Scalability
~~~~~~~~~~

- Use horizontal scaling for the API
- Use a load balancer
- Configure resource limits and requests
- Use auto-scaling

Reliability
~~~~~~~~~

- Set up monitoring and alerting
- Configure health checks
- Implement backup and restore procedures
- Use a distributed database
- Set up high availability

Performance
~~~~~~~~~~

- Use caching
- Optimize database queries
- Use hardware acceleration
- Configure connection pooling

Monitoring
~~~~~~~~~

- Set up logging
- Configure metrics collection
- Set up distributed tracing
- Create dashboards
- Configure alerts

Deployment Checklist
------------------

Before deploying Feluda to production, ensure that:

1. All tests pass
2. Security scans pass
3. Performance benchmarks meet requirements
4. Documentation is up to date
5. Backup and restore procedures are tested
6. Monitoring and alerting are configured
7. Scaling and high availability are tested
8. Rollback procedures are tested
