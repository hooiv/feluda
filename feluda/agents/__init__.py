"""
Agents module for Feluda.

This module provides autonomous agents for the Feluda framework.
"""

from feluda.agents.agent import (
    Agent,
    AgentAction,
    AgentGoal,
    AgentManager,
    AgentObservation,
    AgentState,
    LearningAgent,
    RuleBasedAgent,
    get_agent_manager,
)
from feluda.agents.communication import (
    BroadcastChannel,
    Channel,
    CommunicationManager,
    DirectChannel,
    Message,
    MessageType,
    TopicChannel,
    get_communication_manager,
)
from feluda.agents.learning import (
    Experience,
    ExperienceBuffer,
    LearningAlgorithm,
    LearningManager,
    LearningModel,
    QLearningModel,
    SarsaModel,
    get_learning_manager,
)
from feluda.agents.swarm import (
    AgentSwarm,
    SwarmManager,
    SwarmMessage,
    SwarmState,
    SwarmTask,
    get_swarm_manager,
)

__all__ = [
    "Agent",
    "AgentAction",
    "AgentGoal",
    "AgentManager",
    "AgentObservation",
    "AgentState",
    "AgentSwarm",
    "BroadcastChannel",
    "Channel",
    "CommunicationManager",
    "DirectChannel",
    "Experience",
    "ExperienceBuffer",
    "LearningAgent",
    "LearningAlgorithm",
    "LearningManager",
    "LearningModel",
    "Message",
    "MessageType",
    "QLearningModel",
    "RuleBasedAgent",
    "SarsaModel",
    "SwarmManager",
    "SwarmMessage",
    "SwarmState",
    "SwarmTask",
    "TopicChannel",
    "get_agent_manager",
    "get_communication_manager",
    "get_learning_manager",
    "get_swarm_manager",
]
