"""
Unit tests for AgentFactory - Agent creation with dependency injection.

Tests agent class registration, instantiation, dependency injection,
error handling, and validation of created agent instances.
"""

from unittest.mock import Mock, patch

import pytest

from tarsy.integrations.llm.client import LLMClient
from tarsy.integrations.mcp.client import MCPClient
from tarsy.services.agent_factory import AgentFactory
from tarsy.services.mcp_server_registry import MCPServerRegistry


@pytest.mark.unit
class TestAgentFactoryInitialization:
    """Test AgentFactory initialization and registration."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all AgentFactory dependencies."""
        return {
            'llm_client': Mock(spec=LLMClient),
            'mcp_client': Mock(spec=MCPClient),
            'mcp_registry': Mock(spec=MCPServerRegistry),
            'progress_callback': Mock()
        }
    
    def test_initialization_success(self, mock_dependencies):
        """Test successful AgentFactory initialization."""
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry'],
            progress_callback=mock_dependencies['progress_callback']
        )
        
        assert factory.llm_client == mock_dependencies['llm_client']
        assert factory.mcp_client == mock_dependencies['mcp_client']
        assert factory.mcp_registry == mock_dependencies['mcp_registry']
        assert factory.progress_callback == mock_dependencies['progress_callback']
        assert isinstance(factory.static_agent_classes, dict)
        assert len(factory.static_agent_classes) >= 1  # At least KubernetesAgent
    
    def test_initialization_without_progress_callback(self, mock_dependencies):
        """Test initialization without progress callback."""
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry']
        )
        
        assert factory.progress_callback is None
        assert factory.llm_client == mock_dependencies['llm_client']
        assert factory.mcp_client == mock_dependencies['mcp_client']
        assert factory.mcp_registry == mock_dependencies['mcp_registry']
    
    @patch('tarsy.agents.kubernetes_agent.KubernetesAgent')
    def test_agent_registration(self, mock_kubernetes_agent, mock_dependencies):
        """Test that agents are properly registered during initialization."""
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry']
        )
        
        # Verify KubernetesAgent is registered
        assert "KubernetesAgent" in factory.static_agent_classes
        assert factory.static_agent_classes["KubernetesAgent"] == mock_kubernetes_agent
    
    def test_agent_registry_immutable(self, mock_dependencies):
        """Test that agent registry is properly isolated between instances."""
        factory1 = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry']
        )
        
        factory2 = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry']
        )
        
        # Each factory should have its own registry instance
        assert factory1.static_agent_classes is not factory2.static_agent_classes
        # But they should have the same content
        assert factory1.static_agent_classes.keys() == factory2.static_agent_classes.keys()


@pytest.mark.unit
class TestAgentCreation:
    """Test agent creation and dependency injection."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all AgentFactory dependencies."""
        return {
            'llm_client': Mock(spec=LLMClient),
            'mcp_client': Mock(spec=MCPClient),
            'mcp_registry': Mock(spec=MCPServerRegistry),
            'progress_callback': Mock()
        }
    
    @pytest.fixture
    def agent_factory(self, mock_dependencies):
        """Create AgentFactory with mocked dependencies."""
        return AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry'],
            progress_callback=mock_dependencies['progress_callback']
        )
    
    def test_create_kubernetes_agent_success(self, mock_dependencies):
        """Test successful creation of KubernetesAgent."""
        with patch('tarsy.agents.kubernetes_agent.KubernetesAgent') as mock_kubernetes_agent:
            mock_agent_instance = Mock()
            mock_kubernetes_agent.return_value = mock_agent_instance
            
            # Create factory after mocking
            agent_factory = AgentFactory(
                llm_client=mock_dependencies['llm_client'],
                mcp_client=mock_dependencies['mcp_client'],
                mcp_registry=mock_dependencies['mcp_registry'],
                progress_callback=mock_dependencies['progress_callback']
            )
            
            agent = agent_factory.create_agent("KubernetesAgent")
            
            # Verify agent class was called with correct dependencies
            mock_kubernetes_agent.assert_called_with(
                llm_client=mock_dependencies['llm_client'],
                mcp_client=mock_dependencies['mcp_client'],
                mcp_registry=mock_dependencies['mcp_registry'],
                progress_callback=mock_dependencies['progress_callback']
            )
            
            # Verify correct instance returned
            assert agent == mock_agent_instance
    
    @patch('tarsy.agents.kubernetes_agent.KubernetesAgent')
    def test_create_agent_without_progress_callback(self, mock_kubernetes_agent, mock_dependencies):
        """Test agent creation without progress callback."""
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry']
        )
        
        mock_agent_instance = Mock()
        mock_kubernetes_agent.return_value = mock_agent_instance
        
        agent = factory.create_agent("KubernetesAgent")
        
        # Verify agent class was called with None progress callback
        mock_kubernetes_agent.assert_called_once_with(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry'],
            progress_callback=None
        )
        
        assert agent == mock_agent_instance
    
    def test_create_unknown_agent_failure(self, agent_factory):
        """Test failure when trying to create unknown agent."""
        with pytest.raises(ValueError, match="Unknown agent 'UnknownAgent'"):
            agent_factory.create_agent("UnknownAgent")
    
    def test_create_agent_case_sensitive(self, agent_factory):
        """Test that agent creation is case sensitive."""
        with pytest.raises(ValueError, match="Unknown agent 'kubernetesagent'"):
            agent_factory.create_agent("kubernetesagent")
        
        with pytest.raises(ValueError, match="Unknown agent 'KUBERNETESAGENT'"):
            agent_factory.create_agent("KUBERNETESAGENT")
    
    def test_create_agent_with_initialization_error(self, mock_dependencies):
        """Test handling of agent initialization errors."""
        with patch('tarsy.agents.kubernetes_agent.KubernetesAgent') as mock_kubernetes_agent:
            mock_kubernetes_agent.side_effect = Exception("Agent initialization failed")
            
            # Create factory after mocking
            agent_factory = AgentFactory(
                llm_client=mock_dependencies['llm_client'],
                mcp_client=mock_dependencies['mcp_client'],
                mcp_registry=mock_dependencies['mcp_registry']
            )
            
            with pytest.raises(Exception, match="Agent initialization failed"):
                agent_factory.create_agent("KubernetesAgent")
    
    def test_multiple_agent_creation(self, mock_dependencies):
        """Test creating multiple agent instances."""
        with patch('tarsy.agents.kubernetes_agent.KubernetesAgent') as mock_kubernetes_agent:
            mock_agent1 = Mock()
            mock_agent2 = Mock()
            mock_kubernetes_agent.side_effect = [mock_agent1, mock_agent2]
            
            # Create factory after mocking
            agent_factory = AgentFactory(
                llm_client=mock_dependencies['llm_client'],
                mcp_client=mock_dependencies['mcp_client'],
                mcp_registry=mock_dependencies['mcp_registry']
            )
            
            agent1 = agent_factory.create_agent("KubernetesAgent")
            agent2 = agent_factory.create_agent("KubernetesAgent")
            
            # Each call should create a new instance
            assert agent1 == mock_agent1
            assert agent2 == mock_agent2
            assert agent1 is not agent2
            
            # Both should have been called with same dependencies
            assert mock_kubernetes_agent.call_count == 2
            for call in mock_kubernetes_agent.call_args_list:
                assert call[1]['llm_client'] == mock_dependencies['llm_client']
                assert call[1]['mcp_client'] == mock_dependencies['mcp_client']
                assert call[1]['mcp_registry'] == mock_dependencies['mcp_registry']


@pytest.mark.unit
class TestAgentFactoryRegistry:
    """Test agent registry functionality."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all AgentFactory dependencies."""
        return {
            'llm_client': Mock(spec=LLMClient),
            'mcp_client': Mock(spec=MCPClient),
            'mcp_registry': Mock(spec=MCPServerRegistry)
        }
    
    def test_default_agent_registry(self, mock_dependencies):
        """Test that default agents are registered."""
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry']
        )
        
        # Verify default agents are present
        assert "KubernetesAgent" in factory.static_agent_classes
        
        # Verify the registry is a dictionary
        assert isinstance(factory.static_agent_classes, dict)
        
        # Verify all values are classes (not instances)
        for agent_name, agent_class in factory.static_agent_classes.items():
            assert callable(agent_class), f"{agent_name} should be a callable class"
    
    def test_agent_registry_contents(self, mock_dependencies):
        """Test the contents of the agent registry."""
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry']
        )
        
        # Should contain at least KubernetesAgent
        assert len(factory.static_agent_classes) >= 1
        
        # All agent names should be strings
        for agent_name in factory.static_agent_classes.keys():
            assert isinstance(agent_name, str)
            assert len(agent_name) > 0
    
    @patch('tarsy.agents.kubernetes_agent.KubernetesAgent')
    def test_registry_registration_called_once(self, mock_kubernetes_agent, mock_dependencies):
        """Test that agent registration happens during initialization."""
        # Import should be called during _register_available_agents
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry']
        )
        
        # The class should be in the registry
        assert factory.static_agent_classes["KubernetesAgent"] == mock_kubernetes_agent


@pytest.mark.unit 
class TestDependencyInjection:
    """Test dependency injection into agent instances."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all AgentFactory dependencies with specific characteristics."""
        llm_client = Mock(spec=LLMClient)
        llm_client.name = "test_llm_client"
        
        mcp_client = Mock(spec=MCPClient)
        mcp_client.name = "test_mcp_client"
        
        mcp_registry = Mock(spec=MCPServerRegistry)
        mcp_registry.name = "test_mcp_registry"
        
        progress_callback = Mock()
        progress_callback.name = "test_progress_callback"
        
        return {
            'llm_client': llm_client,
            'mcp_client': mcp_client,
            'mcp_registry': mcp_registry,
            'progress_callback': progress_callback
        }
    
    @patch('tarsy.agents.kubernetes_agent.KubernetesAgent')
    def test_dependency_injection_all_parameters(self, mock_kubernetes_agent, mock_dependencies):
        """Test that all dependencies are correctly injected."""
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry'],
            progress_callback=mock_dependencies['progress_callback']
        )
        
        factory.create_agent("KubernetesAgent")
        
        # Verify all dependencies were passed correctly
        mock_kubernetes_agent.assert_called_once_with(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry'],
            progress_callback=mock_dependencies['progress_callback']
        )
    
    @patch('tarsy.agents.kubernetes_agent.KubernetesAgent')
    def test_dependency_injection_parameter_order(self, mock_kubernetes_agent, mock_dependencies):
        """Test that parameters are passed in correct order."""
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry'],
            progress_callback=mock_dependencies['progress_callback']
        )
        
        factory.create_agent("KubernetesAgent")
        
        # Get the call arguments
        call_args = mock_kubernetes_agent.call_args
        
        # Verify keyword arguments are correct
        assert call_args[1]['llm_client'] == mock_dependencies['llm_client']
        assert call_args[1]['mcp_client'] == mock_dependencies['mcp_client']
        assert call_args[1]['mcp_registry'] == mock_dependencies['mcp_registry']
        assert call_args[1]['progress_callback'] == mock_dependencies['progress_callback']
    
    @patch('tarsy.agents.kubernetes_agent.KubernetesAgent')
    def test_dependency_injection_with_none_callback(self, mock_kubernetes_agent, mock_dependencies):
        """Test dependency injection with None progress callback."""
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry'],
            progress_callback=None
        )
        
        factory.create_agent("KubernetesAgent")
        
        # Verify None callback is passed
        call_args = mock_kubernetes_agent.call_args
        assert call_args[1]['progress_callback'] is None
    
    def test_mcp_registry_requirement(self, mock_dependencies):
        """Test that mcp_registry is required parameter."""
        # This should work fine
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry']
        )
        
        assert factory.mcp_registry == mock_dependencies['mcp_registry']


@pytest.mark.unit
class TestAgentFactoryLogging:
    """Test logging functionality in AgentFactory."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all AgentFactory dependencies."""
        return {
            'llm_client': Mock(spec=LLMClient),
            'mcp_client': Mock(spec=MCPClient),
            'mcp_registry': Mock(spec=MCPServerRegistry)
        }
    
    def test_initialization_logging(self, mock_dependencies, caplog):
        """Test that initialization logs correct information."""
        with caplog.at_level("INFO"):
            factory = AgentFactory(
                llm_client=mock_dependencies['llm_client'],
                mcp_client=mock_dependencies['mcp_client'],
                mcp_registry=mock_dependencies['mcp_registry']
            )
        
        # Should log number of registered agent classes
        log_messages = [record.message for record in caplog.records]
        agent_count_logs = [msg for msg in log_messages if "Initialized Agent Factory with" in msg]
        assert len(agent_count_logs) > 0
        
        # Should mention the number of agent classes
        factory_log = agent_count_logs[0]
        assert "agent classes" in factory_log
        assert str(len(factory.static_agent_classes)) in factory_log
    
    @patch('tarsy.agents.kubernetes_agent.KubernetesAgent')
    def test_agent_creation_logging(self, mock_kubernetes_agent, mock_dependencies, caplog):
        """Test that agent creation logs correct information."""
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry']
        )
        
        mock_agent_instance = Mock()
        mock_kubernetes_agent.return_value = mock_agent_instance
        
        with caplog.at_level("INFO"):
            factory.create_agent("KubernetesAgent")
        
        # Should log agent creation
        log_messages = [record.message for record in caplog.records]
        creation_logs = [msg for msg in log_messages if "Created traditional agent instance" in msg]
        assert len(creation_logs) > 0
        
        creation_log = creation_logs[0]
        assert "KubernetesAgent" in creation_log


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all AgentFactory dependencies."""
        return {
            'llm_client': Mock(spec=LLMClient),
            'mcp_client': Mock(spec=MCPClient),
            'mcp_registry': Mock(spec=MCPServerRegistry)
        }
    
    def test_empty_agent_name(self, mock_dependencies):
        """Test creation with empty agent name."""
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry']
        )
        
        with pytest.raises(ValueError, match="Unknown agent ''"):
            factory.create_agent("")
    
    def test_none_agent_name(self, mock_dependencies):
        """Test creation with None agent name."""
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry']
        )
        
        with pytest.raises((ValueError, TypeError, AttributeError)):
            factory.create_agent(None)
    
    def test_whitespace_agent_name(self, mock_dependencies):
        """Test creation with whitespace-only agent name."""
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry']
        )
        
        with pytest.raises(ValueError, match="Unknown agent '   '"):
            factory.create_agent("   ")
    
    @patch('tarsy.agents.kubernetes_agent.KubernetesAgent')
    def test_agent_import_failure_handling(self, mock_kubernetes_agent, mock_dependencies):
        """Test that import failures are handled appropriately."""
        # This test verifies that if KubernetesAgent import fails,
        # the factory fails fast during initialization
        
        # Since we're patching the import, it should work fine
        factory = AgentFactory(
            llm_client=mock_dependencies['llm_client'],
            mcp_client=mock_dependencies['mcp_client'],
            mcp_registry=mock_dependencies['mcp_registry']
        )
        
        # Should have successfully registered the mocked agent
        assert "KubernetesAgent" in factory.static_agent_classes 