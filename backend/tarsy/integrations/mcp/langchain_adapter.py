"""
MCP to LangChain tool adapter for TARSy.

This module bridges TARSy's existing MCP infrastructure with LangChain's create_agent
by using the official langchain-mcp-adapters package and wrapping tools with TARSy's
security and observability features.
"""

from typing import Any, List, Optional

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools

from tarsy.integrations.mcp.client import MCPClient
from tarsy.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class TarsyMCPToLangChainAdapter:
    """
    Adapter to convert TARSy MCP sessions to LangChain tools.
    
    This adapter uses the official langchain-mcp-adapters package to convert
    MCP tools, then wraps them with TARSy's data masking and audit trail features.
    This approach provides:
    - Battle-tested tool conversion from langchain-mcp-adapters
    - TARSy's security (data masking)
    - TARSy's observability (hooks and audit trail)
    - Reduced maintenance burden (less custom code)
    """
    
    def __init__(self, mcp_client: MCPClient):
        """
        Initialize the adapter with TARSy's MCP client.
        
        Args:
            mcp_client: TARSy's MCPClient instance with initialized sessions
        """
        self.mcp_client = mcp_client
        self.data_masking_service = mcp_client.data_masking_service
    
    async def get_tools_from_servers(
        self, 
        server_ids: List[str],
        session_id: str,
        stage_execution_id: Optional[str] = None
    ) -> List[StructuredTool]:
        """
        Convert MCP servers to LangChain-compatible tools with TARSy features.
        
        This method:
        1. Uses langchain-mcp-adapters to load tools from MCP sessions
        2. Wraps them with TARSy's data masking and audit trail
        3. Returns tools ready for use with create_agent
        
        Args:
            server_ids: List of MCP server IDs to load tools from
            session_id: Session ID for tracking tool calls
            stage_execution_id: Optional stage execution ID
            
        Returns:
            List of LangChain StructuredTool instances with TARSy features
        """
        all_tools = []
        
        for server_id in server_ids:
            logger.debug(f"Loading tools from MCP server: {server_id}")
            
            # Get MCP session for this server
            session = self.mcp_client.sessions.get(server_id)
            if not session:
                logger.warning(f"MCP server '{server_id}' not initialized, skipping")
                continue
            
            try:
                # Use official langchain-mcp-adapters to load tools
                tools = await load_mcp_tools(session)
                logger.info(f"Loaded {len(tools)} tools from server '{server_id}' via langchain-mcp-adapters")
                
                # Wrap each tool with TARSy's security and observability features
                for tool in tools:
                    wrapped_tool = self._wrap_with_tarsy_features(
                        tool=tool,
                        server_id=server_id,
                        session_id=session_id,
                        stage_execution_id=stage_execution_id
                    )
                    all_tools.append(wrapped_tool)
                    
            except Exception as e:
                logger.error(f"Failed to load tools from server '{server_id}': {e}")
                continue
        
        logger.info(f"Total LangChain tools created: {len(all_tools)}")
        return all_tools
    
    def _wrap_with_tarsy_features(
        self,
        tool: StructuredTool,
        server_id: str,
        session_id: str,
        stage_execution_id: Optional[str]
    ) -> StructuredTool:
        """
        Wrap a LangChain tool with TARSy's data masking and audit trail features.
        
        This ensures all tool calls go through TARSy's MCP client, which provides:
        - Data masking for sensitive information
        - Hook integration for audit trail
        - Token counting and cost tracking
        - Summarization for large results
        
        Args:
            tool: Original LangChain tool from langchain-mcp-adapters
            server_id: ID of the MCP server this tool belongs to
            session_id: Session ID for tracking
            stage_execution_id: Optional stage execution ID
            
        Returns:
            Wrapped LangChain StructuredTool with TARSy features
        """
        original_name = tool.name
        tool_description = tool.description or f"Tool {original_name} from {server_id}"
        
        # Apply TARSy namespacing (server.tool format)
        namespaced_name = f"{server_id}.{original_name}"
        
        # Create wrapped async function that routes through TARSy's MCP client
        async def wrapped_func(**kwargs: Any) -> str:
            """Execute tool through TARSy's MCP client for security and observability."""
            try:
                logger.debug(f"Executing tool: {namespaced_name} with args: {kwargs}")
                
                # Call through TARSy's MCP client to get:
                # - Data masking
                # - Hook integration (audit trail)
                # - Token counting
                # - Summarization for large results
                result = await self.mcp_client.call_tool(
                    server_name=server_id,
                    tool_name=original_name,
                    parameters=kwargs,
                    session_id=session_id,
                    stage_execution_id=stage_execution_id
                )
                
                # Extract result content
                result_str = result.get("result", str(result))
                
                logger.debug(f"Tool {namespaced_name} completed successfully")
                return result_str
                
            except Exception as e:
                error_msg = f"Error executing {namespaced_name}: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        # Create new StructuredTool with TARSy features
        wrapped_tool = StructuredTool(
            name=namespaced_name,
            description=tool_description,
            coroutine=wrapped_func,
            args_schema=tool.args_schema  # Preserve original schema from langchain-mcp-adapters
        )
        
        logger.debug(f"Wrapped tool with TARSy features: {namespaced_name}")
        return wrapped_tool

