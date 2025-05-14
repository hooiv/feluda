Changelog
=========

This page documents the changes in each release of Feluda.

v1.0.0 (Unreleased)
------------------

Major Features
~~~~~~~~~~~~~

* Formally verified core components with contract programming
* Zero-trust secure and verifiable lifecycle
* AI-driven autonomous agents (experimental)
* Autonomic (self-tuning/healing/optimizing) systems
* Hardware acceleration hooks
* Advanced cryptographic hooks

Infrastructure & Tooling
~~~~~~~~~~~~~~~~~~~~~~

* Configured Hatch workspace with precisely pinned, hash-verified dependencies
* Set up devcontainer.json for consistent development environment
* Implemented zero-trust secure CI/CD pipeline with comprehensive testing and security checks
* Added SBOM generation and attestation
* Integrated Sigstore for artifact signing
* Implemented automated rollback trigger

Core Enhancements
~~~~~~~~~~~~~~~

* Implemented BaseFeludaOperator abstract base class with contract enforcement
* Created comprehensive exception hierarchy
* Implemented versioned Pydantic models for data structures
* Added formal verification hooks for critical components
* Integrated runtime contract checking with deal

Testing Enhancements
~~~~~~~~~~~~~~~~~

* Migrated tests to pytest
* Implemented hypothesis contract testing
* Added mutation testing with mutmut
* Integrated CrossHair for static contract verification
* Added chaos testing fixtures
* Implemented grammar-based fuzzing
* Added differential fuzzing
* Implemented metamorphic testing
* Added symbolic execution POC

Performance Enhancements
~~~~~~~~~~~~~~~~~~~~~

* Profiled all operators
* Applied Numba JIT compilation
* Developed Rust bindings POC
* Implemented hardware acceleration hooks
* Added hardware-specific profiles
* Designed FPGA/ASIC backend hooks

Autonomic Systems
~~~~~~~~~~~~~~

* Implemented Circuit Breaker pattern
* Added ML-driven performance tuning
* Implemented ML-driven healing

Advanced Cryptography
~~~~~~~~~~~~~~~~~

* Designed Zero-Knowledge Proof hooks
* Designed Homomorphic Encryption hooks
* Designed Secure Multi-Party Computation hooks

Observability
~~~~~~~~~~

* Implemented structlog for structured logging
* Added OpenTelemetry tracing
* Implemented metrics hooks

Documentation
~~~~~~~~~~

* Set up Sphinx documentation
* Added comprehensive API reference
* Created guides for advanced features
* Added tutorials and examples

v0.9.4 (2023-XX-XX)
------------------

* Updated readme instructions
* Fixed docker-compose file issues
* Updated and pinned packages to work with cp311, fixes from `pip-audit`
* Recreated `requirements.txt`

v0.9.3 (2023-XX-XX)
------------------

* Added documentation for tesseract OCR operator
* Created privacy policy

v0.9.2 (2023-XX-XX)
------------------

* Updated README.md
* Added todo to convert an image search model into an operator
* Added test for end-to-end index endpoint
* Refactored to separate feluda core code and user code
* Added overview, operators, architecture documentation
* Added Gatsby site for documentation

v0.9.1 (2023-XX-XX)
------------------

* Added tqdm and updated requirements.txt
* Added image fingerprint with each upload_image call
* Added support for doc_id for upload_text and upload_image APIs
* Added text detection in uploaded image
* Fixed image_upload API
