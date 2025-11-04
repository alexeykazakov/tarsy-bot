"""
LangChain v1.0 proof-of-concept agent implementation.

This agent uses LangChain's create_agent instead of TARSy's custom ReAct controllers,
demonstrating the next-generation agent architecture while maintaining compatibility
with TARSy's ChainContext and AgentExecutionResult interfaces.
"""

from typing import Dict, List, Optional, Any
import json

from langchain.agents import create_agent

from tarsy.agents.langchain_v1.middleware import (
    TarsyLoggingMiddleware,
    StageContextMiddleware,
    TokenCountingMiddleware
)
from tarsy.integrations.llm.client import LLMManager
from tarsy.integrations.mcp.client import MCPClient
from tarsy.integrations.mcp.langchain_adapter import TarsyMCPToLangChainAdapter
from tarsy.models.agent_execution_result import AgentExecutionResult
from tarsy.models.constants import StageStatus
from tarsy.models.processing_context import ChainContext
from tarsy.services.mcp_server_registry import MCPServerRegistry
from tarsy.utils.logger import get_module_logger
from tarsy.utils.timestamp import now_us

logger = get_module_logger(__name__)


class LangChainV1Agent:
    """
    Proof-of-concept agent using LangChain v1.0's create_agent.
    
    This agent duplicates KubernetesAgent functionality but uses LangChain's
    built-in ReAct implementation instead of TARSy's custom controllers.
    
    Key differences from BaseAgent:
    - No inheritance from BaseAgent (parallel implementation)
    - Uses create_agent for ReAct loop instead of iteration controllers
    - Leverages middleware instead of custom hooks
    - Maintains compatibility with ChainContext/AgentExecutionResult
    """
    
    def __init__(
        self,
        llm_client: LLMManager,
        mcp_client: MCPClient,
        mcp_registry: MCPServerRegistry
    ):
        """
        Initialize LangChain v1.0 agent.
        
        Args:
            llm_client: TARSy's LLM manager for accessing configured providers
            mcp_client: TARSy's MCP client with initialized server sessions
            mcp_registry: Registry of MCP server configurations
        """
        self.llm_client = llm_client
        self.mcp_client = mcp_client
        self.mcp_registry = mcp_registry
        self._langchain_agent = None
        self._adapter = TarsyMCPToLangChainAdapter(mcp_client)
        self._current_stage_execution_id: Optional[str] = None
        
        logger.info("Initialized LangChainV1Agent (PoC)")
    
    def mcp_servers(self) -> List[str]:
        """
        Return the MCP server IDs required for this agent.
        
        Same as KubernetesAgent - uses kubernetes-server.
        
        Returns:
            List containing kubernetes-server ID
        """
        return ["kubernetes-server"]
    
    def custom_instructions(self) -> str:
        """
        Return agent-specific custom instructions.
        
        Returns:
            Custom instructions for Kubernetes analysis
        """
        return """You are a Kubernetes troubleshooting expert"""
    
    def set_current_stage_execution_id(self, stage_execution_id: str) -> None:
        """
        Set the current stage execution ID for interaction tagging.
        
        Args:
            stage_execution_id: Stage execution ID from database
        """
        self._current_stage_execution_id = stage_execution_id
        logger.debug(f"Set stage execution ID: {stage_execution_id}")
    
    def get_current_stage_execution_id(self) -> Optional[str]:
        """
        Get the current stage execution ID.
        
        Returns:
            Current stage execution ID or None
        """
        return self._current_stage_execution_id
    
    async def initialize_agent(
        self, 
        session_id: str,
        chain_context: Optional[ChainContext] = None
    ) -> None:
        """
        Initialize create_agent with MCP tools and middleware.
        
        This method:
        1. Loads MCP tools via the adapter
        2. Creates LangChain agent with middleware
        3. Caches the agent for reuse
        
        Args:
            session_id: Session ID for tool execution tracking
            chain_context: Optional chain context for stage context middleware
        """
        logger.info("Initializing LangChain v1.0 agent...")
        
        # Get MCP tools via adapter
        tools = await self._adapter.get_tools_from_servers(
            server_ids=self.mcp_servers(),
            session_id=session_id,
            stage_execution_id=self._current_stage_execution_id
        )
        
        logger.info(f"Loaded {len(tools)} LangChain tools from MCP servers")
        
        # Build system prompt
        system_prompt = self._build_system_prompt()
        
        # Get the actual configured LangChain model from TARSy's LLM client
        # This ensures we use TARSy's API key, temperature, and other settings
        client = self.llm_client.get_client()
        if not client or not client.llm_client:
            raise ValueError("No configured LLM client available")
        
        langchain_model = client.llm_client  # This is the actual BaseChatModel
        
        # Create middleware stack
        middleware = [
            TarsyLoggingMiddleware(
                session_id=session_id,
                stage_execution_id=self._current_stage_execution_id,
                system_prompt=system_prompt  # Pass system prompt for conversation context
            ),
            StageContextMiddleware(chain_context=chain_context),
            TokenCountingMiddleware()
        ]
        
        # Create agent using LangChain v1.0 with TARSy's configured model
        self._langchain_agent = create_agent(
            model=langchain_model,  # Pass the actual configured model, not a string
            tools=tools,
            system_prompt=system_prompt,
            middleware=middleware
        )
        
        logger.info("LangChain v1.0 agent initialized successfully")
    
    async def process_alert(self, chain_context: ChainContext) -> AgentExecutionResult:
        """
        Process alert using LangChain's create_agent.
        
        This is the main entry point called by AlertService.
        
        Args:
            chain_context: ChainContext containing alert and stage information
            
        Returns:
            AgentExecutionResult with analysis findings
        """
        try:
            logger.info(f"Processing alert with LangChainV1Agent for session {chain_context.session_id}")
            
            # Initialize agent if not already done
            if not self._langchain_agent:
                await self.initialize_agent(
                    session_id=chain_context.session_id,
                    chain_context=chain_context
                )
            
            # Build messages for LangChain agent
            messages = self._build_messages_from_context(chain_context)
            
            # Invoke LangChain agent (middleware handles logging)
            logger.debug("Invoking LangChain agent...")
            result = await self._langchain_agent.ainvoke(
                {"messages": messages},
                config={
                    "recursion_limit": 20  # Allow up to 20 steps for thorough investigation
                }
            )
            
            # Convert LangChain result to AgentExecutionResult
            return self._convert_to_agent_result(result, chain_context)
            
        except Exception as e:
            error_msg = f"LangChainV1Agent processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return AgentExecutionResult(
                status=StageStatus.FAILED,
                agent_name=self.__class__.__name__,
                timestamp_us=now_us(),
                result_summary=error_msg,
                error_message=error_msg
            )
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt from instructions.
        
        Follows TARSy's three-tier instruction pattern:
        1. General SRE instructions
        2. MCP server-specific instructions
        3. Custom agent instructions
        
        Returns:
            Complete system prompt
        """
        instructions = []
        
        # General instructions
        instructions.append("You are an expert SRE (Site Reliability Engineer) assistant.")
        instructions.append("Your role is to analyze alerts and provide actionable insights.")
        instructions.append("\n## Investigation Guidelines")
        instructions.append("- Conduct THOROUGH investigations using multiple tools")
        instructions.append("- Gather ALL relevant information before concluding")
        instructions.append("- Use tools step-by-step to build a complete picture")
        instructions.append("- Don't stop after just one tool call - investigate deeply")
        instructions.append("- Verify findings with additional checks when needed")
        instructions.append("\n## CRITICAL: Tool Usage Rules")
        instructions.append("- ALWAYS use the EXACT tool names provided (including server prefix)")
        instructions.append("- Tool names are in the format: server-name.tool-name")
        instructions.append("- Example: Use 'kubernetes-server.resources_get', NOT 'resources_get'")
        instructions.append("- If a tool fails with 'not a valid tool' error, CHECK the available tool list")
        instructions.append("- If you keep getting tool errors, STOP and report what tools you tried")
        
        # MCP server instructions
        for server_id in self.mcp_servers():
            server_config = self.mcp_registry.get_server_config_safe(server_id)
            if server_config and server_config.instructions:
                instructions.append(f"\n## Instructions for {server_id}")
                instructions.append(server_config.instructions)
        
        # Custom instructions
        custom = self.custom_instructions()
        if custom:
            instructions.append("\n## Agent-Specific Instructions")
            instructions.append(custom)
        
        return "\n\n".join(instructions)
    
    def _build_messages_from_context(self, chain_context: ChainContext) -> List[Dict[str, str]]:
        """
        Build LangChain messages from ChainContext.
        
        Args:
            chain_context: TARSy ChainContext
            
        Returns:
            List of message dicts for LangChain
        """
        messages = []
        
        # Add user message with alert data
        alert_data = chain_context.processing_alert.alert_data
        
        # Build user prompt
        user_content = f"""# Alert Analysis Request

## Alert Data
{json.dumps(alert_data, indent=2)}

## Runbook
{chain_context.get_runbook_content() or "No runbook available"}

## Previous Stage Results
{self._format_previous_stages(chain_context)}

Please analyze this alert and provide:
1. Your analysis plan
2. Investigation findings
3. Root cause analysis
4. Recommended actions
"""
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def _format_previous_stages(self, chain_context: ChainContext) -> str:
        """
        Format previous stage results for inclusion in prompt.
        
        Args:
            chain_context: Chain context with stage results
            
        Returns:
            Formatted string of previous results
        """
        previous_results = chain_context.get_previous_stages_results()
        
        if not previous_results:
            return "This is the first stage."
        
        formatted = []
        for stage_name, result in previous_results:
            formatted.append(f"### {stage_name}")
            formatted.append(result.final_analysis or result.result_summary)
        
        return "\n\n".join(formatted)
    
    def _convert_to_agent_result(
        self, 
        langchain_result: Dict[str, Any],
        chain_context: ChainContext
    ) -> AgentExecutionResult:
        """
        Convert LangChain agent result to TARSy's AgentExecutionResult.
        
        Args:
            langchain_result: Result from create_agent.ainvoke()
            chain_context: Chain context for metadata
            
        Returns:
            AgentExecutionResult compatible with TARSy infrastructure
        """
        # Extract messages from LangChain result
        messages = langchain_result.get("messages", [])
        
        # Get final AI message
        final_response = ""
        for msg in reversed(messages):
            if hasattr(msg, 'content') and hasattr(msg, 'type'):
                if msg.type == 'ai':
                    # Handle both string content and structured content blocks
                    content = msg.content
                    if isinstance(content, str):
                        final_response = content
                    elif isinstance(content, list):
                        # New LangChain format: list of content blocks
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_parts.append(block.get('text', ''))
                        final_response = '\n'.join(text_parts)
                    else:
                        final_response = str(content)
                    break
        
        if not final_response:
            final_response = "No response generated"
        
        # Build result summary (include conversation history for transparency)
        result_summary = self._build_result_summary(messages)
        
        return AgentExecutionResult(
            status=StageStatus.COMPLETED,
            agent_name=self.__class__.__name__,
            stage_name=chain_context.current_stage_name,
            stage_description="Kubernetes analysis using LangChain v1.0",
            timestamp_us=now_us(),
            result_summary=result_summary,
            complete_conversation_history=result_summary,  # Same for now
            final_analysis=final_response
        )
    
    def _build_result_summary(self, messages: List[Any]) -> str:
        """
        Build result summary from LangChain messages.
        
        Args:
            messages: List of LangChain messages
            
        Returns:
            Formatted summary string
        """
        summary_parts = []
        
        for msg in messages:
            if not hasattr(msg, 'type') or not hasattr(msg, 'content'):
                continue
            
            msg_type = msg.type
            raw_content = msg.content
            
            # Extract text from structured content blocks
            if isinstance(raw_content, str):
                content = raw_content
            elif isinstance(raw_content, list):
                # New LangChain format: list of content blocks
                text_parts = []
                for block in raw_content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text_parts.append(block.get('text', ''))
                content = '\n'.join(text_parts)
            else:
                content = str(raw_content)
            
            if msg_type == 'human':
                summary_parts.append(f"**User:**\n{content}\n")
            elif msg_type == 'ai':
                summary_parts.append(f"**Assistant:**\n{content}\n")
            elif msg_type == 'tool':
                summary_parts.append(f"**Tool Result:**\n{content}\n")
        
        return "\n".join(summary_parts)