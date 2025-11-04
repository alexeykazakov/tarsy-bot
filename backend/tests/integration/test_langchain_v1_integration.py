"""
Integration tests for LangChainV1Agent.

These tests verify end-to-end functionality with real MCP server integration
(mocked at the transport level).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tarsy.agents.langchain_v1_agent import LangChainV1Agent
from tarsy.config.settings import Settings
from tarsy.integrations.llm.client import LLMClient
from tarsy.integrations.mcp.client import MCPClient
from tarsy.models.agent_execution_result import AgentExecutionResult
from tarsy.models.constants import StageStatus
from tarsy.models.llm_models import LLMProviderConfig
from tarsy.models.processing_context import ChainContext, ProcessingAlert
from tarsy.models.alert_processing import AlertKey, AlertData
from tarsy.services.mcp_server_registry import MCPServerRegistry


@pytest.mark.integration
@pytest.mark.asyncio
class TestLangChainV1AgentIntegration:
    """Integration test suite for LangChainV1Agent."""
    
    @pytest.fixture
    async def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock(spec=Settings)
        settings.llm_provider = "openai"
        settings.llm_providers = {
            "openai": LLMProviderConfig(
                type="openai",
                model="gpt-4o-mini",
                api_key="test-key",
                temperature=0.7
            )
        }
        return settings
    
    @pytest.fixture
    async def mcp_client_with_session(self, mock_settings):
        """Create MCP client with mocked session."""
        registry = MCPServerRegistry()
        client = MCPClient(settings=mock_settings, mcp_registry=registry)
        
        # Mock session
        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        mock_session.call_tool = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text="Tool result")]
        ))
        
        client.sessions["kubernetes-server"] = mock_session
        client._initialized = True
        
        return client
    
    @pytest.fixture
    def sample_alert_context(self):
        """Create sample alert context."""
        processing_alert = ProcessingAlert(
            alert_key=AlertKey(
                alert_type="kubernetes-v1-test",
                alert_name="pod-crash-loop",
                namespace="production"
            ),
            alert_data=AlertData(
                alert_type="kubernetes-v1-test",
                alert_name="pod-crash-loop",
                namespace="production",
                data={
                    "pod_name": "my-app-7d9f8c5b6-xk2ln",
                    "container": "app",
                    "restart_count": 5,
                    "last_state": "CrashLoopBackOff"
                }
            ),
            matched_chain_id="langchain-v1-kubernetes-chain",
            runbook_url=None
        )
        
        return ChainContext.from_processing_alert(
            processing_alert=processing_alert,
            session_id="integration-test-session",
            current_stage_name="analysis"
        )
    
    async def test_full_agent_execution_flow(
        self,
        mock_settings,
        mcp_client_with_session,
        sample_alert_context
    ):
        """Test complete agent execution flow."""
        # Create LLM client (mocked to avoid real API calls)
        llm_client = MagicMock(spec=LLMClient)
        llm_client.provider_config = mock_settings.llm_providers["openai"]
        llm_client.model = "gpt-4o-mini"
        llm_client.available = True
        
        # Create registry
        registry = MCPServerRegistry()
        
        # Create agent
        agent = LangChainV1Agent(
            llm_client=llm_client,
            mcp_client=mcp_client_with_session,
            mcp_registry=registry
        )
        
        # Set stage execution ID
        agent.set_current_stage_execution_id("test-stage-123")
        
        # Mock the LangChain agent creation
        with patch('tarsy.agents.langchain_v1_agent.create_agent') as mock_create:
            # Mock agent response
            mock_agent_instance = AsyncMock()
            mock_ai_msg = MagicMock()
            mock_ai_msg.type = "ai"
            mock_ai_msg.content = """# Analysis Results

## Investigation Plan
1. Check pod status
2. Review container logs
3. Analyze restart patterns

## Findings
The pod is experiencing CrashLoopBackOff with 5 restarts.

## Recommendations
1. Review application logs
2. Check resource limits
3. Verify configuration"""
            
            mock_agent_instance.ainvoke = AsyncMock(return_value={
                "messages": [mock_ai_msg]
            })
            
            mock_create.return_value = mock_agent_instance
            
            # Process alert
            result = await agent.process_alert(sample_alert_context)
        
        # Verify result structure
        assert isinstance(result, AgentExecutionResult)
        assert result.status == StageStatus.COMPLETED
        assert result.agent_name == "LangChainV1Agent"
        assert result.stage_name == "analysis"
        assert len(result.final_analysis) > 0
        
        # Verify content
        assert "Investigation Plan" in result.final_analysis or "Findings" in result.final_analysis
    
    async def test_agent_with_mcp_tools(
        self,
        mock_settings,
        mcp_client_with_session,
        sample_alert_context
    ):
        """Test agent with MCP tools loaded."""
        # Create LLM client
        llm_client = MagicMock(spec=LLMClient)
        llm_client.provider_config = mock_settings.llm_providers["openai"]
        llm_client.model = "gpt-4o-mini"
        
        # Add mock tools to MCP session
        mock_tool = MagicMock()
        mock_tool.name = "kubernetes_get_pod"
        mock_tool.description = "Get pod details"
        
        mock_session = mcp_client_with_session.sessions["kubernetes-server"]
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[mock_tool]
        ))
        
        # Create registry and agent
        registry = MCPServerRegistry()
        agent = LangChainV1Agent(
            llm_client=llm_client,
            mcp_client=mcp_client_with_session,
            mcp_registry=registry
        )
        
        # Initialize agent and verify tools are loaded
        await agent.initialize_agent(
            session_id=sample_alert_context.session_id,
            chain_context=sample_alert_context
        )
        
        # Verify agent was initialized
        assert agent._langchain_agent is not None


