"""
Session Operations for Session Data Service.

Handles core session lifecycle management including creation, status updates,
and retrieval operations.
"""

import logging
from typing import Optional

from tarsy.models.agent_config import ChainConfigModel
from tarsy.models.constants import AlertSessionStatus
from tarsy.models.db_models import AlertSession
from tarsy.models.processing_context import ChainContext
from tarsy.services.session_data.base_infrastructure import BaseSessionDataInfra
from tarsy.utils.timestamp import now_us

logger = logging.getLogger(__name__)


class SessionOperations:
    """
    Session lifecycle operations.
    
    Manages creation, updates, and basic queries for alert processing sessions.
    """
    
    def __init__(self, infra: BaseSessionDataInfra):
        """
        Initialize session operations with shared infrastructure.
        
        Args:
            infra: Shared infrastructure instance
        """
        self._infra = infra
    
    def create_session(
        self,
        chain_context: ChainContext,
        chain_definition: ChainConfigModel
    ) -> bool:
        """
        Create a new alert processing session with retry logic.
        
        Args:
            chain_context: Chain processing context containing session data
            chain_definition: Chain configuration model with chain details
            
        Returns:
            True if created successfully, False if failed
        """
        def _create_session_operation():
            with self._infra.get_repository() as repo:
                if not repo:
                    raise RuntimeError("Repository unavailable - cannot create session")
                
                # Extract data from ChainContext and ChainConfigModel
                agent_type = f"chain:{chain_definition.chain_id}"  # Construct agent_type internally
                
                session = AlertSession(
                    session_id=chain_context.session_id,
                    alert_data=chain_context.processing_alert.alert_data,
                    agent_type=agent_type,
                    alert_type=chain_context.processing_alert.alert_type,
                    status=AlertSessionStatus.PENDING.value,
                    chain_id=chain_definition.chain_id,
                    chain_definition=chain_definition.model_dump(),  # Store as JSON-serializable dict
                    author=chain_context.author,  # User who submitted the alert (from oauth2-proxy headers)
                    runbook_url=chain_context.processing_alert.runbook_url,  # Runbook URL for re-submission
                    mcp_selection=chain_context.mcp.model_dump() if chain_context.mcp else None  # MCP selection for re-submission
                )
                
                created_session = repo.create_alert_session(session)
                if created_session:
                    logger.info(f"Created session {created_session.session_id}")
                    return True
                return False
        
        result = self._infra._retry_database_operation("create_session", _create_session_operation)
        return result if result is not None else False
    
    def update_session_status(
        self,
        session_id: str,
        status: str,
        error_message: Optional[str] = None,
        final_analysis: Optional[str] = None,
        final_analysis_summary: Optional[str] = None,
        executive_summary_error: Optional[str] = None,
        pause_metadata: Optional[dict] = None
    ) -> bool:
        """
        Update session processing status with retry logic.
        
        Args:
            session_id: The session identifier
            status: New status (pending, in_progress, completed, failed, paused)
            error_message: Error message if status is failed
            final_analysis: Final formatted analysis if status is completed
            final_analysis_summary: Executive summary of final analysis if generated successfully
            executive_summary_error: Error message if executive summary generation failed
            pause_metadata: Pause metadata if status is paused
            
        Returns:
            True if updated successfully, False otherwise
        """
        if not session_id:
            return False
        
        def _update_status_operation():
            with self._infra.get_repository() as repo:
                if not repo:
                    raise RuntimeError("Repository unavailable - cannot update session status")
                
                session = repo.get_alert_session(session_id)
                if not session:
                    logger.warning(f"Session {session_id} not found for status update")
                    return False
                
                session.status = status
                if error_message:
                    session.error_message = error_message
                if final_analysis:
                    session.final_analysis = final_analysis
                if final_analysis_summary:
                    session.final_analysis_summary = final_analysis_summary
                if executive_summary_error:
                    session.executive_summary_error = executive_summary_error
                # Set pause_metadata when transitioning to PAUSED, clear it otherwise
                if status == AlertSessionStatus.PAUSED.value:
                    session.pause_metadata = pause_metadata
                else:
                    # Clear pause_metadata when not paused (keep it clean)
                    session.pause_metadata = None
                if status in AlertSessionStatus.terminal_values():
                    session.completed_at_us = now_us()
                
                success = repo.update_alert_session(session)
                if success:
                    logger.debug(f"Updated session {session_id} status to {status}")
                        
                return success
        
        result = self._infra._retry_database_operation("update_session_status", _update_status_operation)
        return result if result is not None else False
    
    def get_session(self, session_id: str) -> Optional['AlertSession']:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            AlertSession if found, None otherwise
        """
        if not session_id:
            return None
        
        def _get_operation():
            with self._infra.get_repository() as repo:
                if not repo:
                    return None
                return repo.get_alert_session(session_id)
        
        return self._infra._retry_database_operation(
            "get_session",
            _get_operation,
            treat_none_as_success=True
        )
    
    def update_session_to_canceling(self, session_id: str) -> tuple[bool, str]:
        """
        Atomically update session to CANCELING if not already terminal.
        
        This is used for session cancellation to ensure we don't cancel
        sessions that have already completed or failed.
        
        Args:
            session_id: Session identifier
            
        Returns:
            (success, current_status): True if updated to CANCELING, False if already terminal.
                                        Also returns the current status.
        """
        if not session_id:
            return (False, "unknown")
        
        def _update_operation():
            with self._infra.get_repository() as repo:
                if not repo:
                    raise RuntimeError("Repository unavailable")
                
                session = repo.get_alert_session(session_id)
                if not session:
                    return (False, "not_found")
                
                # Check if already terminal - don't update if so
                if session.status in AlertSessionStatus.terminal_values():
                    logger.info(f"Session {session_id} already terminal: {session.status}")
                    return (False, session.status)
                
                # Check if already canceling - idempotent
                if session.status == AlertSessionStatus.CANCELING.value:
                    logger.info(f"Session {session_id} already canceling")
                    return (True, session.status)
                
                # Update to CANCELING
                session.status = AlertSessionStatus.CANCELING.value
                success = repo.update_alert_session(session)
                
                if success:
                    logger.info(f"Updated session {session_id} to CANCELING")
                    return (True, AlertSessionStatus.CANCELING.value)
                
                return (False, session.status)
        
        result = self._infra._retry_database_operation("update_to_canceling", _update_operation)
        return result if result else (False, "error")
