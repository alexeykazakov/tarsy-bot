"""
Custom middleware for integrating LangChain v1.0 agents with TARSy infrastructure.

This module provides middleware that bridges LangChain's agent system with
TARSy's hooks, stage context, and observability features.
"""

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from langchain_core.messages import AIMessage, BaseMessage

from tarsy.hooks.hook_context import llm_interaction_context
from tarsy.models.agent_execution_result import AgentExecutionResult
from tarsy.models.processing_context import ChainContext
from tarsy.models.unified_interactions import LLMConversation, LLMMessage, MessageRole
from tarsy.utils.logger import get_module_logger

logger = get_module_logger(__name__)
llm_logger = get_module_logger("llm.communications")


class TarsyLoggingMiddleware(AgentMiddleware):
    """
    Middleware that integrates with TARSy's hook system for LLM interactions.
    
    This middleware captures LLM requests and responses, integrating them with
    TARSy's existing audit trail and observability infrastructure using LangChain's
    wrap_model_call hook.
    """
    
    def __init__(
        self, 
        session_id: str, 
        stage_execution_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize logging middleware.
        
        Args:
            session_id: Session ID for tracking
            stage_execution_id: Optional stage execution ID
            system_prompt: System prompt used by the agent (needed for conversation context)
        """
        self.session_id = session_id
        self.stage_execution_id = stage_execution_id
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.request_counter = 0
        logger.debug(
            f"Initialized TarsyLoggingMiddleware for session {session_id}, "
            f"stage {stage_execution_id}"
        )
    
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """
        Wrap LLM calls to log interactions in TARSy format AND persist to database.
        
        This hook is called around each LLM call, allowing us to log both
        the request and response for TARSy's audit trail and persist them
        to the database via TARSy's hook system.
        
        Args:
            request: Model request with state and messages
            handler: Async callback that executes the model
            
        Returns:
            Model response
        """
        self.request_counter += 1
        request_id = f"llm_req_{self.request_counter}"
        
        # Extract messages, model info, and tools
        messages = request.state.get("messages", [])
        model_name = getattr(request.model, 'model_name', 'unknown') if hasattr(request, 'model') else 'unknown'
        
        # Tools can be in multiple places - check all of them
        tools = []
        if hasattr(request, 'tools') and request.tools:
            tools = request.tools
        elif request.state.get("tools"):
            tools = request.state.get("tools", [])
        elif hasattr(request.model, 'bound_tools') and request.model.bound_tools:
            tools = request.model.bound_tools
        
        # Build request data for hook context
        request_data = {
            'messages': [{'role': getattr(msg, 'type', 'user'), 'content': str(msg.content)[:500]} 
                        for msg in messages],
            'model': model_name,
            'provider': 'langchain',  # LangChain v1.0 agent
            'temperature': 0.7,  # Default, actual temp comes from model config
            'tools': len(tools) if tools else 0  # Number of tools available
        }
        
        # Debug: Log what's in the request object
        logger.debug(f"Request object attributes: {dir(request)}")
        logger.debug(f"Request.state keys: {request.state.keys() if hasattr(request, 'state') else 'N/A'}")
        if hasattr(request, 'model'):
            logger.debug(f"Model attributes: {dir(request.model)}")
        
        # Log available tools
        if tools:
            llm_logger.info(f"LLM Request [ID: {request_id}] has {len(tools)} tools available")
            llm_logger.debug(f"Tools: {[getattr(t, 'name', 'unknown') for t in tools]}")
        else:
            llm_logger.warning(f"LLM Request [ID: {request_id}] has NO tools captured (checking request.tools, request.state.tools, and model.bound_tools)")
        
        # Log request preview
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'content'):
                prompt_preview = str(last_msg.content)[:200]
                llm_logger.info(f"LLM Request [ID: {request_id}] (prompt preview): {prompt_preview}...")
        
        # Use TARSy's hook context to persist to database
        async with llm_interaction_context(
            self.session_id, 
            request_data, 
            self.stage_execution_id
        ) as ctx:
            # Execute the model
            response = await handler(request)
            
            # Extract response content and build conversation
            response_content = None
            if isinstance(response, ModelResponse) and response.result:
                for msg in response.result:
                    if isinstance(msg, AIMessage):
                        content = msg.content
                        if isinstance(content, str):
                            response_content = content
                            response_preview = content[:200]
                        elif isinstance(content, list):
                            # Extract text from content blocks
                            text_parts = []
                            for block in content:
                                if isinstance(block, dict) and block.get('type') == 'text':
                                    text_parts.append(block.get('text', ''))
                            response_content = '\n'.join(text_parts)
                            response_preview = response_content[:200]
                        else:
                            response_content = str(content)
                            response_preview = response_content[:200]
                        
                        llm_logger.info(f"LLM Response [ID: {request_id}] (preview): {response_preview}...")
                        break
            
            # Capture LLM reasoning and communication
            # Focus on the reasoning process, not low-level debugging
            try:
                import json
                
                # Build a user-friendly view of the LLM's reasoning
                reasoning_data = {
                    "reasoning": None,  # Internal thought process
                    "response": None,   # Final response
                    "tool_calls": [],   # Actions the LLM wants to take
                    "usage": {},        # Token usage
                    "_raw_interaction": {  # Complete technical data for debugging
                        "request_to_llm": {},
                        "response_from_llm": {}
                    }
                }
                
                # Extract REASONING from the LLM response (user-friendly)
                try:
                    if response.result:
                        for msg in response.result:
                            # Only process AI messages (skip human/tool messages)
                            if getattr(msg, 'type', None) != 'ai':
                                continue
                            
                            # Try to use content_blocks (LangChain v1.0 unified API)
                            if hasattr(msg, 'content_blocks'):
                                for block in msg.content_blocks:
                                    if isinstance(block, dict):
                                        if block.get("type") == "reasoning":
                                            reasoning_data["reasoning"] = block.get("reasoning")
                                        elif block.get("type") == "text":
                                            reasoning_data["response"] = block.get("text")
                            
                            # Fallback: Extract from content directly
                            if not reasoning_data["response"] and hasattr(msg, 'content'):
                                content = msg.content
                                if isinstance(content, str):
                                    reasoning_data["response"] = content
                                elif isinstance(content, list):
                                    # Extract text blocks
                                    text_parts = []
                                    for block in content:
                                        if isinstance(block, dict) and block.get('type') == 'text':
                                            text_parts.append(block.get('text', ''))
                                    reasoning_data["response"] = '\n'.join(text_parts)
                            
                            # Extract tool calls
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    reasoning_data["tool_calls"].append({
                                        "tool": tool_call.get('name'),
                                        "arguments": tool_call.get('args'),
                                        "id": tool_call.get('id')
                                    })
                            
                            # Extract token usage
                            if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                                usage = msg.usage_metadata
                                if hasattr(usage, 'model_dump'):
                                    reasoning_data["usage"] = usage.model_dump()
                                elif hasattr(usage, 'dict'):
                                    reasoning_data["usage"] = usage.dict()
                                elif isinstance(usage, dict):
                                    reasoning_data["usage"] = usage
                
                except Exception as reasoning_err:
                    logger.warning(f"Failed to extract reasoning: {reasoning_err}")
                
                # Also capture RAW interaction for debugging
                try:
                    reasoning_data["_raw_interaction"]["request_to_llm"] = {
                        "messages": [msg.model_dump() if hasattr(msg, 'model_dump') else str(msg) for msg in messages],
                        "model": model_name,
                    }
                    if response.result:
                        reasoning_data["_raw_interaction"]["response_from_llm"] = {
                            "messages": [msg.model_dump() if hasattr(msg, 'model_dump') else str(msg) for msg in response.result]
                        }
                except Exception:
                    pass  # Raw data is optional
                
                # Format as user-friendly JSON
                reasoning_json = json.dumps(reasoning_data, indent=2, default=str, ensure_ascii=False)
                
                # Store as a single message
                tarsy_messages = [
                    LLMMessage(
                        role=MessageRole.SYSTEM, 
                        content=f"=== LLM REASONING & RESPONSE ===\n\n{reasoning_json}"
                    )
                ]
                
                logger.debug(f"Captured LLM reasoning ({len(reasoning_json)} chars)")
                conversation = LLMConversation(messages=tarsy_messages)
                ctx.interaction.conversation = conversation
                    
            except Exception as e:
                logger.warning(f"Failed to capture LLM reasoning: {e}", exc_info=True)
            
            # Trigger hooks to persist to database
            await ctx.complete_success({})
        
        return response


class StageContextMiddleware(AgentMiddleware):
    """
    Middleware that injects previous stage results into agent context.
    
    This middleware enables multi-stage agent chains by providing agents with
    context about what previous stages discovered, allowing for progressive
    data enrichment.
    """
    
    def __init__(self, chain_context: Optional[ChainContext] = None):
        """
        Initialize stage context middleware.
        
        Args:
            chain_context: ChainContext containing previous stage results
        """
        self.chain_context = chain_context
        logger.debug("Initialized StageContextMiddleware")
        # Note: Stage context injection will be implemented in a future iteration
        # The correct approach is to modify the system prompt before agent creation
        # rather than trying to inject via middleware
    
    def _build_stage_context_summary(
        self, stage_results: List[Tuple[str, AgentExecutionResult]]
    ) -> str:
        """
        Build a summary of previous stage results.
        
        Args:
            stage_results: List of (stage_name, AgentExecutionResult) tuples
            
        Returns:
            Formatted summary string
        """
        if not stage_results:
            return ""
        
        summary_parts = []
        for stage_name, result in stage_results:
            # Extract key information from AgentExecutionResult
            summary = f"### Stage: {stage_name}\n"
            
            if hasattr(result, 'analysis'):
                summary += f"{result.analysis}\n"
            elif hasattr(result, 'result'):
                summary += f"{result.result}\n"
            else:
                summary += f"{str(result)}\n"
            
            summary_parts.append(summary)
        
        return "\n\n".join(summary_parts)


class TokenCountingMiddleware(AgentMiddleware):
    """
    Middleware for tracking token usage across agent execution.
    
    This middleware could be enhanced to integrate with TARSy's token counting
    and cost tracking infrastructure.
    """
    
    def __init__(self):
        """Initialize token counting middleware."""
        self.total_tokens = 0
        logger.debug("Initialized TokenCountingMiddleware")
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """
        Track token usage by wrapping the model call.
        
        Args:
            request: Model request
            handler: Async callback that executes the model
            
        Returns:
            Model response
        """
        # Execute the model
        response = await handler(request)
        
        # Extract token usage from response if available
        if isinstance(response, ModelResponse) and response.result:
            # Check the AI message for usage metadata
            for msg in response.result:
                if isinstance(msg, AIMessage) and hasattr(msg, 'usage_metadata'):
                    usage = msg.usage_metadata
                    if isinstance(usage, dict):
                        tokens = usage.get('total_tokens', 0)
                        self.total_tokens += tokens
                        logger.debug(f"Token usage: {tokens} (total: {self.total_tokens})")
                        break
        
        return response
    
    def get_total_tokens(self) -> int:
        """
        Get total token count for this agent execution.
        
        Returns:
            Total tokens used
        """
        return self.total_tokens