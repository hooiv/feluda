"""
Deployment manager for Feluda.

This module provides a deployment manager for model serving.
"""

from feluda.deployment.manager import (
    Deployment,
    DeploymentManager,
    DeploymentStatus,
    get_deployment_manager,
)
from feluda.deployment.serving import (
    ModelServer,
    ModelServerBackend,
    FlaskModelServer,
    FastAPIModelServer,
    TorchServeModelServer,
    get_model_server,
)
from feluda.deployment.scaling import (
    ScalingPolicy,
    AutoScaler,
    get_auto_scaler,
)
from feluda.deployment.monitoring import (
    DeploymentMonitor,
    get_deployment_monitor,
)

__all__ = [
    "AutoScaler",
    "Deployment",
    "DeploymentManager",
    "DeploymentMonitor",
    "DeploymentStatus",
    "FastAPIModelServer",
    "FlaskModelServer",
    "ModelServer",
    "ModelServerBackend",
    "ScalingPolicy",
    "TorchServeModelServer",
    "get_auto_scaler",
    "get_deployment_manager",
    "get_deployment_monitor",
    "get_model_server",
]
