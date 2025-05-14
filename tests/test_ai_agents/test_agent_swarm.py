"""
Unit tests for the AI agent swarm module.
"""

import unittest
from unittest import mock

import pytest

from feluda.ai_agents.agent_swarm import (
    Agent,
    AgentRole,
    AgentSwarm,
    Message,
    SwarmConfig,
    create_development_swarm,
)


class TestAgentSwarm(unittest.TestCase):
    """Test cases for the agent swarm module."""
    
    def test_message(self):
        """Test the Message class."""
        # Create a message
        message = Message(
            sender="agent1",
            receiver="agent2",
            content="Hello, world!",
            metadata={"key": "value"},
        )
        
        # Check the attributes
        self.assertEqual(message.sender, "agent1")
        self.assertEqual(message.receiver, "agent2")
        self.assertEqual(message.content, "Hello, world!")
        self.assertEqual(message.metadata, {"key": "value"})
        
        # Test to_dict
        message_dict = message.to_dict()
        self.assertEqual(message_dict["sender"], "agent1")
        self.assertEqual(message_dict["receiver"], "agent2")
        self.assertEqual(message_dict["content"], "Hello, world!")
        self.assertEqual(message_dict["metadata"], {"key": "value"})
        
        # Test from_dict
        message2 = Message.from_dict(message_dict)
        self.assertEqual(message2.sender, "agent1")
        self.assertEqual(message2.receiver, "agent2")
        self.assertEqual(message2.content, "Hello, world!")
        self.assertEqual(message2.metadata, {"key": "value"})
    
    def test_agent(self):
        """Test the Agent class."""
        # Create an agent
        agent = Agent(
            name="agent1",
            role=AgentRole.DEVELOPER,
            model="gpt-4",
            api_key="test_key",
            api_url="https://api.example.com",
        )
        
        # Check the attributes
        self.assertEqual(agent.name, "agent1")
        self.assertEqual(agent.role, AgentRole.DEVELOPER)
        self.assertEqual(agent.model, "gpt-4")
        self.assertEqual(agent.api_key, "test_key")
        self.assertEqual(agent.api_url, "https://api.example.com")
        
        # Test send_message
        with mock.patch("feluda.ai_agents.agent_swarm.Agent._call_api") as mock_call_api:
            mock_call_api.return_value = "Response"
            
            response = agent.send_message("Hello, world!")
            
            mock_call_api.assert_called_once()
            self.assertEqual(response, "Response")
    
    def test_swarm_config(self):
        """Test the SwarmConfig class."""
        # Create a config
        config = SwarmConfig(
            agents=[
                {
                    "name": "agent1",
                    "role": AgentRole.DEVELOPER,
                    "model": "gpt-4",
                },
                {
                    "name": "agent2",
                    "role": AgentRole.REVIEWER,
                    "model": "gpt-4",
                },
            ],
            api_key="test_key",
            api_url="https://api.example.com",
            max_steps=10,
            timeout=60.0,
        )
        
        # Check the attributes
        self.assertEqual(len(config.agents), 2)
        self.assertEqual(config.agents[0]["name"], "agent1")
        self.assertEqual(config.agents[0]["role"], AgentRole.DEVELOPER)
        self.assertEqual(config.agents[1]["name"], "agent2")
        self.assertEqual(config.agents[1]["role"], AgentRole.REVIEWER)
        self.assertEqual(config.api_key, "test_key")
        self.assertEqual(config.api_url, "https://api.example.com")
        self.assertEqual(config.max_steps, 10)
        self.assertEqual(config.timeout, 60.0)
        
        # Test to_dict
        config_dict = config.to_dict()
        self.assertEqual(len(config_dict["agents"]), 2)
        self.assertEqual(config_dict["agents"][0]["name"], "agent1")
        self.assertEqual(config_dict["agents"][0]["role"], AgentRole.DEVELOPER)
        self.assertEqual(config_dict["api_key"], "test_key")
        self.assertEqual(config_dict["api_url"], "https://api.example.com")
        self.assertEqual(config_dict["max_steps"], 10)
        self.assertEqual(config_dict["timeout"], 60.0)
        
        # Test from_dict
        config2 = SwarmConfig.from_dict(config_dict)
        self.assertEqual(len(config2.agents), 2)
        self.assertEqual(config2.agents[0]["name"], "agent1")
        self.assertEqual(config2.agents[0]["role"], AgentRole.DEVELOPER)
        self.assertEqual(config2.api_key, "test_key")
        self.assertEqual(config2.api_url, "https://api.example.com")
        self.assertEqual(config2.max_steps, 10)
        self.assertEqual(config2.timeout, 60.0)
    
    def test_agent_swarm(self):
        """Test the AgentSwarm class."""
        # Create a swarm
        config = SwarmConfig(
            agents=[
                {
                    "name": "agent1",
                    "role": AgentRole.DEVELOPER,
                    "model": "gpt-4",
                },
                {
                    "name": "agent2",
                    "role": AgentRole.REVIEWER,
                    "model": "gpt-4",
                },
            ],
            api_key="test_key",
            api_url="https://api.example.com",
            max_steps=10,
            timeout=60.0,
        )
        
        swarm = AgentSwarm(config)
        
        # Check the attributes
        self.assertEqual(len(swarm.agents), 2)
        self.assertEqual(swarm.agents[0].name, "agent1")
        self.assertEqual(swarm.agents[0].role, AgentRole.DEVELOPER)
        self.assertEqual(swarm.agents[1].name, "agent2")
        self.assertEqual(swarm.agents[1].role, AgentRole.REVIEWER)
        self.assertEqual(swarm.config.api_key, "test_key")
        self.assertEqual(swarm.config.api_url, "https://api.example.com")
        self.assertEqual(swarm.config.max_steps, 10)
        self.assertEqual(swarm.config.timeout, 60.0)
        
        # Test run
        with mock.patch("feluda.ai_agents.agent_swarm.Agent.send_message") as mock_send_message:
            mock_send_message.return_value = "Response"
            
            result = swarm.run(task="Test task", steps=1)
            
            mock_send_message.assert_called_once()
            self.assertIsInstance(result, dict)
            self.assertIn("messages", result)
            self.assertIn("result", result)
    
    def test_create_development_swarm(self):
        """Test the create_development_swarm function."""
        # Create a swarm
        with mock.patch("feluda.ai_agents.agent_swarm.AgentSwarm") as mock_swarm:
            mock_swarm_instance = mock.MagicMock()
            mock_swarm.return_value = mock_swarm_instance
            
            swarm = create_development_swarm(
                task="Test task",
                code_context="Test context",
                api_key="test_key",
                api_url="https://api.example.com",
            )
            
            mock_swarm.assert_called_once()
            self.assertEqual(swarm, mock_swarm_instance)


if __name__ == "__main__":
    unittest.main()
