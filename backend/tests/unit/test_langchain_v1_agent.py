"""
Unit tests for LangChainV1Agent.

Tests the LangChain v1.0 PoC agent functionality including MCP adapter,
middleware, and result conversion.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from tarsy.agents.langchain_v1_agent import LangChainV1Agent
from tarsy.integrations.mcp.langchain_adapter import TarsyMCPToLangChainAdapter
from tarsy.models.agent_execution_result import AgentExecutionResult
from tarsy.models.constants import StageStatus
from tarsy.models.processing_context import ChainContext, ProcessingAlert
from tarsy.models.alert_processing import AlertKey, AlertData


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()
    client.provider_config = MagicMock()
    client.provider_config.type = "openai"
    client.model = "gpt-4o"
    return client


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client with sessions."""
    client = MagicMock()
    client.sessions = {"kubernetes-server": MagicMock()}
    client.data_masking_service = MagicMock()
    client.call_tool = AsyncMock(return_value={"result": "Tool executed successfully"})
    return client


@pytest.fixture
def mock_mcp_registry():
    """Create a mock MCP server registry."""
    registry = MagicMock()
    server_config = MagicMock()
    server_config.instructions = "Test MCP instructions"
    registry.get_server_config_safe = MagicMock(return_value=server_config)
    return registry


@pytest.fixture
def sample_chain_context():
    """Create a sample ChainContext for testing."""
    processing_alert = ProcessingAlert(
        alert_key=AlertKey(
            alert_type="kubernetes-v1-test",
            alert_name="test-alert",
            namespace="test-namespace"
        ),
        alert_data=AlertData(
            alert_type="kubernetes-v1-test",
            alert_name="test-alert",
            namespace="test-namespace",
            data={"status": "pending"}
        ),
        matched_chain_id="langchain-v1-kubernetes-chain",
        runbook_url=None
    )
    
    context = ChainContext.from_processing_alert(
        processing_alert=processing_alert,
        session_id="test-session-123",
        current_stage_name="analysis"
    )
    
    return context


class TestLangChainV1Agent:
    """Test suite for LangChainV1Agent."""
    
    def test_initialization(self, mock_llm_client, mock_mcp_client, mock_mcp_registry):
        """Test agent initialization."""
        agent = LangChainV1Agent(
            llm_client=mock_llm_client,
            mcp_client=mock_mcp_client,
            mcp_registry=mock_mcp_registry
        )
        
        assert agent.llm_client == mock_llm_client
        assert agent.mcp_client == mock_mcp_client
        assert agent.mcp_registry == mock_mcp_registry
        assert agent._langchain_agent is None
    
    def test_mcp_servers(self, mock_llm_client, mock_mcp_client, mock_mcp_registry):
        """Test mcp_servers returns correct server IDs."""
        agent = LangChainV1Agent(
            mock_llm_client, mock_mcp_client, mock_mcp_registry
        )
        
        servers = agent.mcp_servers()
        assert servers == ["kubernetes-server"]
    
    def test_custom_instructions(self, mock_llm_client, mock_mcp_client, mock_mcp_registry):
        """Test custom_instructions returns expected instructions."""
        agent = LangChainV1Agent(
            mock_llm_client, mock_mcp_client, mock_mcp_registry
        )
        
        instructions = agent.custom_instructions()
        assert "Kubernetes" in instructions
        assert "plan of action" in instructions
    
    def test_get_langchain_model_string(self, mock_llm_client, mock_mcp_client, mock_mcp_registry):
        """Test model string conversion to LangChain format."""
        agent = LangChainV1Agent(
            mock_llm_client, mock_mcp_client, mock_mcp_registry
        )
        
        model_string = agent._get_langchain_model_string()
        assert model_string == "openai:gpt-4o"
    
    def test_build_system_prompt(self, mock_llm_client, mock_mcp_client, mock_mcp_registry):
        """Test system prompt building."""
        agent = LangChainV1Agent(
            mock_llm_client, mock_mcp_client, mock_mcp_registry
        )
        
        prompt = agent._build_system_prompt()
        assert "SRE" in prompt
        assert "Test MCP instructions" in prompt
        assert "Kubernetes" in prompt
    
    def test_build_messages_from_context(
        self, 
        mock_llm_client, 
        mock_mcp_client, 
        mock_mcp_registry,
        sample_chain_context
    ):
        """Test message building from ChainContext."""
        agent = LangChainV1Agent(
            mock_llm_client, mock_mcp_client, mock_mcp_registry
        )
        
        messages = agent._build_messages_from_context(sample_chain_context)
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Alert Data" in messages[0]["content"]
        assert "kubernetes-v1-test" in messages[0]["content"]
    
    @pytest.mark.asyncio
    async def test_process_alert_success(
        self,
        mock_llm_client,
        mock_mcp_client,
        mock_mcp_registry,
        sample_chain_context
    ):
        """Test successful alert processing."""
        agent = LangChainV1Agent(
            mock_llm_client, mock_mcp_client, mock_mcp_registry
        )
        
        # Mock the LangChain agent
        mock_langchain_agent = AsyncMock()
        mock_ai_message = MagicMock()
        mock_ai_message.type = "ai"
        mock_ai_message.content = "Test analysis result"
        
        mock_langchain_agent.ainvoke = AsyncMock(return_value={
            "messages": [mock_ai_message]
        })
        
        agent._langchain_agent = mock_langchain_agent
        
        # Process alert
        result = await agent.process_alert(sample_chain_context)
        
        # Verify result
        assert isinstance(result, AgentExecutionResult)
        assert result.status == StageStatus.COMPLETED
        assert result.agent_name == "LangChainV1Agent"
        assert "Test analysis result" in result.final_analysis
    
    @pytest.mark.asyncio
    async def test_process_alert_failure(
        self,
        mock_llm_client,
        mock_mcp_client,
        mock_mcp_registry,
        sample_chain_context
    ):
        """Test alert processing with failure."""
        agent = LangChainV1Agent(
            mock_llm_client, mock_mcp_client, mock_mcp_registry
        )
        
        # Mock the LangChain agent to raise an exception
        mock_langchain_agent = AsyncMock()
        mock_langchain_agent.ainvoke = AsyncMock(side_effect=Exception("Test error"))
        
        agent._langchain_agent = mock_langchain_agent
        
        # Process alert
        result = await agent.process_alert(sample_chain_context)
        
        # Verify error handling
        assert isinstance(result, AgentExecutionResult)
        assert result.status == StageStatus.FAILED
        assert "Test error" in result.error_message


class TestTarsyMCPToLangChainAdapter:
    """Test suite for TarsyMCPToLangChainAdapter."""
    
    @pytest.mark.asyncio
    async def test_get_tools_from_servers_uses_official_adapter(self, mock_mcp_client):
        """Test tool loading uses official langchain-mcp-adapters."""
        # Mock a LangChain tool from langchain-mcp-adapters
        mock_lc_tool = MagicMock()
        mock_lc_tool.name = "test_tool"
        mock_lc_tool.description = "Test tool description"
        mock_lc_tool.args_schema = None
        
        # Mock session
        mock_session = mock_mcp_client.sessions["kubernetes-server"]
        
        # Create adapter
        adapter = TarsyMCPToLangChainAdapter(mock_mcp_client)
        
        # Patch load_mcp_tools from official adapter
        with patch('tarsy.integrations.mcp.langchain_adapter.load_mcp_tools') as mock_load:
            mock_load.return_value = [mock_lc_tool]
            
            # Get tools
            tools = await adapter.get_tools_from_servers(
                server_ids=["kubernetes-server"],
                session_id="test-session",
                stage_execution_id="test-stage"
            )
            
            # Verify official adapter was called
            mock_load.assert_called_once_with(mock_session)
            
            # Verify tools were wrapped with TARSy features
            assert len(tools) == 1
            assert tools[0].name == "kubernetes-server.test_tool"
            assert tools[0].description == "Test tool description"
    
    @pytest.mark.asyncio
    async def test_wrapped_tool_routes_through_tarsy_mcp_client(self, mock_mcp_client):
        """Test wrapped tool calls go through TARSy's MCP client for security/observability."""
        # Mock a LangChain tool
        mock_lc_tool = MagicMock()
        mock_lc_tool.name = "get_pod"
        mock_lc_tool.description = "Get pod details"
        mock_lc_tool.args_schema = None
        
        # Mock session
        mock_session = mock_mcp_client.sessions["kubernetes-server"]
        
        # Create adapter
        adapter = TarsyMCPToLangChainAdapter(mock_mcp_client)
        
        # Patch load_mcp_tools
        with patch('tarsy.integrations.mcp.langchain_adapter.load_mcp_tools') as mock_load:
            mock_load.return_value = [mock_lc_tool]
            
            # Get wrapped tools
            tools = await adapter.get_tools_from_servers(
                server_ids=["kubernetes-server"],
                session_id="test-session",
                stage_execution_id="test-stage"
            )
            
            # Execute the wrapped tool
            result = await tools[0].coroutine(namespace="default")
            
            # Verify it called TARSy's MCP client (which provides masking/hooks)
            mock_mcp_client.call_tool.assert_called_once_with(
                server_name="kubernetes-server",
                tool_name="get_pod",
                parameters={"namespace": "default"},
                session_id="test-session",
                stage_execution_id="test-stage"
            )
            
            # Verify result extraction
            assert result == "Tool executed successfully"
    
    @pytest.mark.asyncio
    async def test_get_tools_with_missing_server(self, mock_mcp_client):
        """Test tool loading with missing server."""
        adapter = TarsyMCPToLangChainAdapter(mock_mcp_client)
        
        # Request tools from non-existent server
        tools = await adapter.get_tools_from_servers(
            server_ids=["non-existent-server"],
            session_id="test-session"
        )
        
        # Should return empty list
        assert len(tools) == 0

