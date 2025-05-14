"""
AI Agents Package

This package provides AI agent integration for Feluda.
"""

from feluda.ai_agents.agent_swarm import (
    Agent,
    AgentMessage,
    AgentRole,
    AgentSwarm,
    LLMAgent,
    create_development_swarm,
)
from feluda.ai_agents.qa_agent import PRAnalyzer, QAAgent

__all__ = [
    # Agent swarm
    "Agent",
    "AgentMessage",
    "AgentRole",
    "AgentSwarm",
    "LLMAgent",
    "create_development_swarm",
    
    # QA agent
    "QAAgent",
    "PRAnalyzer",
]
