"""
Conversation Operations for Session Data Service.

Handles conversation history retrieval, formatting, and building comprehensive
session histories for chat follow-ups and external analysis.
"""

import json
import logging
from typing import Final, Optional, Tuple

from tarsy.models.history_models import ConversationMessage, LLMConversationHistory
from tarsy.models.unified_interactions import LLMInteraction
from tarsy.services.session_data.base_infrastructure import BaseSessionDataInfra

logger = logging.getLogger(__name__)

# Sentinel value to indicate no LLM interactions found (distinct from None/failure)
class _NoInteractionsSentinel:
    """Sentinel to distinguish 'no interactions found' from database failures."""
    def __repr__(self) -> str:
        return "<NO_INTERACTIONS>"

NO_INTERACTIONS: Final[_NoInteractionsSentinel] = _NoInteractionsSentinel()


class ConversationOperations:
    """
    Conversation history operations.
    
    Manages conversation history retrieval, formatting for chat contexts,
    and building comprehensive session histories for external analysis.
    """
    
    def __init__(self, infra: BaseSessionDataInfra):
        """
        Initialize conversation operations with shared infrastructure.
        
        Args:
            infra: Shared infrastructure instance
        """
        self._infra = infra
    
    def get_session_conversation_history(
        self,
        session_id: str,
        include_chat: bool = False
    ) -> Tuple[Optional[LLMConversationHistory], Optional[LLMConversationHistory]]:
        """
        Get LLM conversation history for a session and optionally its chat.
        
        Strategy:
        1. For main session: Get the last final_analysis type interaction (or fallback to last by timestamp)
        2. For chat (if include_chat=True): Get the last chat stage's final_analysis interaction
        
        The conversation includes all accumulated messages up to that point,
        providing the full context for analysis/evaluation.
        
        Args:
            session_id: Session identifier
            include_chat: If True, also get chat conversation history
            
        Returns:
            Tuple of (session_conversation, chat_conversation), either can be None
        """
        def _get_conversation_history():
            with self._infra.get_repository() as repo:
                if not repo:
                    return None, None
                
                # Get main session conversation
                session_interaction = repo.get_last_llm_interaction_with_conversation(
                    session_id=session_id,
                    prefer_final_analysis=True,
                    chat_id=None  # Main session, exclude chat stages
                )
                session_conversation = self._build_conversation_history(session_interaction)
                
                # Get chat conversation if requested
                chat_conversation = None
                if include_chat:
                    chat = repo.get_chat_by_session(session_id)
                    if chat:
                        chat_interaction = repo.get_last_llm_interaction_with_conversation(
                            session_id=session_id,
                            prefer_final_analysis=True,
                            chat_id=chat.chat_id
                        )
                        chat_conversation = self._build_conversation_history(chat_interaction)
                
                return session_conversation, chat_conversation
        
        result = self._infra._retry_database_operation(
            "get_session_conversation_history",
            _get_conversation_history
        )
        return result if result else (None, None)
    
    def _build_conversation_history(
        self,
        interaction: Optional[LLMInteraction]
    ) -> Optional[LLMConversationHistory]:
        """
        Build LLMConversationHistory from an LLMInteraction.
        
        Args:
            interaction: LLMInteraction with conversation data
            
        Returns:
            LLMConversationHistory model or None if no valid conversation
        """
        if not interaction or not interaction.conversation:
            return None
        
        try:
            # Convert conversation messages to simple format
            messages = [
                ConversationMessage(
                    role=msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                    content=msg.content
                )
                for msg in interaction.conversation.messages
            ]
            
            return LLMConversationHistory(
                model_name=interaction.model_name,
                provider=interaction.provider,
                timestamp_us=interaction.timestamp_us,
                input_tokens=interaction.input_tokens,
                output_tokens=interaction.output_tokens,
                total_tokens=interaction.total_tokens,
                messages=messages
            )
        except Exception as e:
            logger.error(f"Failed to build conversation history: {str(e)}")
            return None
    
    def get_formatted_session_conversation(
        self,
        session_id: str,
        exclude_chat_stages: bool = True,
        include_thinking: bool = False
    ) -> str:
        """
        Get formatted conversation text for any session (ReAct, Native Thinking, Parallel).
        
        This method works universally across all iteration strategies by operating
        on the LLMConversation structure which is strategy-agnostic. It retrieves
        all relevant interactions and formats them using PromptBuilder, optionally
        including thinking_content from native thinking models.
        
        The formatted output includes:
        - Investigation history header (📋 INVESTIGATION HISTORY)
        - Initial investigation request (alert + runbook + previous stages + tools)
        - Internal reasoning (optional, from native thinking models correlated with responses)
        - All ReAct reasoning or Native Thinking responses
        - Tool observations
        - Final analysis
        
        I can be used by different components, such as:
        - ChatService: To capture session investigation history for follow-up chats (include_thinking=False)
        - ScoringService: As the SESSION_CONVERSATION component for judge prompts (include_thinking=True)
        
        Args:
            session_id: Session identifier
            exclude_chat_stages: If True, only include main investigation stages (default)
            include_thinking: If True, include internal reasoning from native thinking models (default: False)
            
        Returns:
            Formatted conversation text with optional thinking_content. When interactions exist
            but none are valid (e.g., session was cancelled before processing), returns a
            cancellation history object rather than raising an error.
            
        Raises:
            ValueError: If no interactions exist for the session (no LLM interactions recorded at all)
        """
        def _get_formatted_conversation():
            with self._infra.get_repository() as repo:
                if not repo:
                    return None
                
                # Get LLM interactions (with optional chat stage filtering)
                all_interactions = repo.get_llm_interactions_for_session(
                    session_id=session_id,
                    exclude_chat_stages=exclude_chat_stages
                )
                
                if not all_interactions:
                    # Return sentinel instead of raising to preserve error message
                    return NO_INTERACTIONS
                
                # Filter to valid interactions
                from tarsy.models.constants import CHAT_CONTEXT_INTERACTION_TYPES
                
                valid_interactions = [
                    interaction for interaction in all_interactions
                    if (interaction.conversation is not None and 
                        interaction.interaction_type in CHAT_CONTEXT_INTERACTION_TYPES)
                ]
                
                # If no valid interactions, return cancellation message
                if not valid_interactions:
                    from tarsy.agents.prompts.builders import PromptBuilder
                    builder = PromptBuilder()
                    return builder.format_investigation_context(None)
                
                # Get last interaction's conversation (has all accumulated messages)
                last_interaction = valid_interactions[-1]
                
                # Format using PromptBuilder with optional thinking_content
                from tarsy.agents.prompts.builders import PromptBuilder
                builder = PromptBuilder()
                return builder.format_investigation_context(
                    conversation=last_interaction.conversation,
                    interactions=valid_interactions if include_thinking else None,
                    include_thinking=include_thinking
                )
        
        result = self._infra._retry_database_operation(
            "get_formatted_session_conversation",
            _get_formatted_conversation
        )
        
        # Check for sentinel indicating no interactions found
        if result is NO_INTERACTIONS:
            raise ValueError(f"No LLM interactions found for session {session_id}")
        
        if result is None:
            raise ValueError(f"Failed to get formatted conversation for session {session_id}")
        
        return result
    
    def build_comprehensive_session_history(
        self,
        session_id: str,
        include_separate_alert_section: bool = True,
        include_thinking: bool = False
    ) -> str:
        """
        Build comprehensive session history for external analysis (scoring, auditing).
        
        Returns complete session information suitable for LLM-based evaluation:
        - Alert metadata and data (optional separate section for structured access)
        - Complete investigation conversation (includes alert + runbook + all stages)
        
        Note: Runbook content is already embedded in the investigation conversation
        (in the initial prompt), so it's not duplicated in a separate section.
        
        This works universally across:
        - ReAct and Native Thinking iteration strategies
        - Sequential and parallel agent stages  
        - Multiple LLM providers
        
        Args:
            session_id: Session identifier
            include_separate_alert_section: If True, add structured alert metadata 
                at the top for easy LLM access (default: True)
            include_thinking: If True, include internal reasoning from native thinking models (default: False)
            
        Returns:
            Formatted text with optional alert section + investigation history (with optional thinking)
            
        Example output (include_separate_alert_section=True):
            ═══════════════════════════════════════════════════════════════
            ALERT INFORMATION
            ═══════════════════════════════════════════════════════════════
            **Alert Type:** PodCrashLoop
            **Session ID:** abc-123
            **Status:** completed
            
            **Alert Data:**
            ```json
            {
              "pod": "my-pod",
              "namespace": "production"
            }
            ```
            
            [followed by full investigation conversation with 📋 INVESTIGATION HISTORY]
        """
        # Get formatted investigation conversation (includes runbook, optional thinking)
        conversation_text = self.get_formatted_session_conversation(
            session_id,
            include_thinking=include_thinking
        )
        
        # If no separate alert section requested, return just the conversation
        if not include_separate_alert_section:
            return conversation_text
        
        # Get session metadata for structured alert section
        # Import here to avoid circular dependency
        from tarsy.services.session_data.session_operations import SessionOperations
        session_ops = SessionOperations(self._infra)
        session = session_ops.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Build alert section for structured access
        sections = []
        sections.append("=" * 79)
        sections.append("ALERT INFORMATION")
        sections.append("=" * 79)
        sections.append("")
        sections.append(f"**Alert Type:** {session.alert_type}")
        sections.append(f"**Session ID:** {session.session_id}")
        # Convert status enum to value if it's an enum
        status_value = session.status.value if hasattr(session.status, 'value') else session.status
        sections.append(f"**Status:** {status_value}")
        sections.append("")
        sections.append("**Alert Data:**")
        sections.append("```json")
        sections.append(json.dumps(session.alert_data, indent=2))
        sections.append("```")
        sections.append("")
        
        # Append investigation conversation (already includes alert + runbook)
        sections.append(conversation_text)
        
        return "\n".join(sections)
