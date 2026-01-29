"""
Stage Operations for Session Data Service.

Handles stage execution management for chain processing including creation,
updates, and queries for stage executions.
"""

import logging
from typing import List, Optional

from tarsy.models.constants import StageStatus
from tarsy.models.db_models import StageExecution
from tarsy.models.history_models import SessionStats
from tarsy.services.session_data.base_infrastructure import BaseSessionDataInfra
from tarsy.utils.timestamp import now_us

logger = logging.getLogger(__name__)


class StageOperations:
    """
    Stage execution operations.
    
    Manages stage execution lifecycle for chain processing including creation,
    status updates, parallel stages, and paused stage handling.
    """
    
    def __init__(self, infra: BaseSessionDataInfra):
        """
        Initialize stage operations with shared infrastructure.
        
        Args:
            infra: Shared infrastructure instance
        """
        self._infra = infra
    
    async def create_stage_execution(self, stage_execution: StageExecution) -> str:
        """Create a new stage execution record."""
        def _create_stage_operation():
            with self._infra.get_repository() as repo:
                if not repo:
                    raise RuntimeError("Repository unavailable - cannot create stage execution record")
                return repo.create_stage_execution(stage_execution)
        
        result = await self._infra._retry_database_operation_async("create_stage_execution", _create_stage_operation)
        if result is None:
            raise RuntimeError(f"Failed to create stage execution record for stage '{stage_execution.stage_name}'. Chain processing cannot continue without proper stage tracking.")
        return result
    
    async def update_stage_execution(self, stage_execution: StageExecution):
        """Update an existing stage execution record."""
        def _update_stage_operation():
            with self._infra.get_repository() as repo:
                if not repo:
                    raise RuntimeError("Repository unavailable - cannot update stage execution")
                return repo.update_stage_execution(stage_execution)
        
        result = await self._infra._retry_database_operation_async("update_stage_execution", _update_stage_operation)
        return result if result is not None else False
    
    async def update_session_current_stage(
        self, 
        session_id: str, 
        current_stage_index: int, 
        current_stage_id: str
    ):
        """Update the current stage information for a session."""
        def _update_current_stage_operation():
            with self._infra.get_repository() as repo:
                if not repo:
                    raise RuntimeError("Repository unavailable - cannot update session current stage")
                return repo.update_session_current_stage(session_id, current_stage_index, current_stage_id)
        
        result = await self._infra._retry_database_operation_async("update_session_current_stage", _update_current_stage_operation)
        return result if result is not None else False

    async def get_session_summary(self, session_id: str) -> Optional[SessionStats]:
        """
        Get just the summary statistics for a session - returns SessionStats model for controllers.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionStats model or None if session not found
        """
        try:
            def _get_session_summary_operation():
                with self._infra.get_repository() as repo:
                    if not repo:
                        raise RuntimeError("Repository unavailable - cannot retrieve session summary")
                    
                    session_overview = repo.get_session_overview(session_id)
                    if not session_overview:
                        # This is a legitimate case - session doesn't exist, not a system failure
                        return None
                    
                    # Calculate statistics from SessionOverview and return as SessionStats
                    from tarsy.models.history_models import (
                        ChainStatistics,
                        SessionStats,
                    )
                    
                    # Calculate basic counts from the SessionOverview model
                    total_interactions = session_overview.total_interactions
                    llm_interactions = session_overview.llm_interaction_count
                    mcp_communications = session_overview.mcp_communication_count
                    
                    # Calculate duration from session timing
                    total_duration_ms = 0
                    if session_overview.started_at_us and session_overview.completed_at_us:
                        total_duration_ms = (session_overview.completed_at_us - session_overview.started_at_us) // 1000
                    
                    # Calculate token usage aggregations - prefer repository-provided session totals
                    session_input_tokens = 0
                    session_output_tokens = 0
                    session_total_tokens = 0

                    # Check if aggregated token totals are already available in session overview
                    if (session_overview.session_input_tokens is not None and 
                        session_overview.session_output_tokens is not None and 
                        session_overview.session_total_tokens is not None):
                        # Use the already-available aggregated session totals from overview
                        session_input_tokens = session_overview.session_input_tokens
                        session_output_tokens = session_overview.session_output_tokens
                        session_total_tokens = session_overview.session_total_tokens
                    else:
                        # Aggregates not available in overview, get detailed session for fallback calculation
                        detailed_session = repo.get_session_details(session_id)
                        if detailed_session:
                            # Check if aggregated session totals are available in detailed session
                            if (detailed_session.session_input_tokens is not None and 
                                detailed_session.session_output_tokens is not None and 
                                detailed_session.session_total_tokens is not None):
                                # Use the repository-provided aggregated session totals
                                session_input_tokens = detailed_session.session_input_tokens
                                session_output_tokens = detailed_session.session_output_tokens
                                session_total_tokens = detailed_session.session_total_tokens
                            else:
                                # Fall back to manual per-stage summation if aggregates are missing
                                for stage in detailed_session.stages:
                                    if stage.stage_input_tokens:
                                        session_input_tokens += stage.stage_input_tokens
                                    if stage.stage_output_tokens: 
                                        session_output_tokens += stage.stage_output_tokens
                                    if stage.stage_total_tokens:
                                        session_total_tokens += stage.stage_total_tokens
                    
                    # Create chain statistics from SessionOverview
                    chain_stats = ChainStatistics(
                        total_stages=session_overview.total_stages or 0,
                        completed_stages=session_overview.completed_stages or 0,
                        failed_stages=session_overview.failed_stages,
                        stages_by_agent={}  # Not calculated in overview for performance
                    )
                    
                    session_stats = SessionStats(
                        total_interactions=total_interactions,
                        llm_interactions=llm_interactions,
                        mcp_communications=mcp_communications,
                        system_events=0,  # Not tracked in SessionOverview
                        errors_count=1 if session_overview.error_message else 0,
                        total_duration_ms=total_duration_ms,
                        session_input_tokens=session_input_tokens,
                        session_output_tokens=session_output_tokens,
                        session_total_tokens=session_total_tokens,
                        chain_statistics=chain_stats
                    )
                    return session_stats
            
            result = await self._infra._retry_database_operation_async(
                "get_session_summary",
                _get_session_summary_operation,
                treat_none_as_success=True,
            )
            return result
                
        except Exception as e:
            logger.error(f"Failed to get session summary for {session_id}: {str(e)}")
            return None
    
    async def get_stage_execution(self, execution_id: str) -> Optional[StageExecution]:
        """Get a single stage execution by ID."""
        def _get_stage_execution_operation():
            with self._infra.get_repository() as repo:
                if not repo:
                    raise RuntimeError("Repository unavailable - cannot retrieve stage execution")
                return repo.get_stage_execution(execution_id)
        
        result = await self._infra._retry_database_operation_async(
            "get_stage_execution",
            _get_stage_execution_operation,
            treat_none_as_success=True,
        )
        return result
    
    async def get_stage_executions(self, session_id: str) -> List[StageExecution]:
        """Get all stage executions for a session."""
        def _get_stage_executions_operation():
            with self._infra.get_repository() as repo:
                if not repo:
                    raise RuntimeError("Repository unavailable - cannot retrieve stage executions")
                return repo.get_stage_executions_for_session(session_id)
        
        result = await self._infra._retry_database_operation_async(
            "get_stage_executions",
            _get_stage_executions_operation,
            treat_none_as_success=True,
        )
        return result or []
    
    async def get_parallel_stage_children(self, parent_execution_id: str) -> List[StageExecution]:
        """Get all child stage executions for a parallel stage parent."""
        def _get_children_operation():
            with self._infra.get_repository() as repo:
                if not repo:
                    raise RuntimeError("Repository unavailable - cannot retrieve parallel stage children")
                return repo.get_parallel_stage_children(parent_execution_id)
        
        result = await self._infra._retry_database_operation_async(
            "get_parallel_stage_children",
            _get_children_operation,
            treat_none_as_success=True,
        )
        return result or []
    
    async def get_paused_stages(self, session_id: str) -> List[StageExecution]:
        """Get all paused stage executions for a session, including parallel children.
        
        Flattens parent stages and their parallel_executions before filtering
        to ensure paused parallel child stages are included.
        """
        def _get_paused_stages_operation():
            with self._infra.get_repository() as repo:
                if not repo:
                    raise RuntimeError("Repository unavailable - cannot retrieve paused stages")
                # Get all stages for session (parents with parallel_executions attached)
                stages = repo.get_stage_executions_for_session(session_id)
                # Build combined list: each parent + its parallel children
                all_stages: List[StageExecution] = []
                for stage in stages:
                    all_stages.append(stage)
                    # Include parallel child stages if present
                    parallel_children = getattr(stage, 'parallel_executions', None)
                    if parallel_children:
                        all_stages.extend(parallel_children)
                # Filter for paused stages (both parents and children)
                return [s for s in all_stages if s.status == StageStatus.PAUSED.value]
        
        result = await self._infra._retry_database_operation_async(
            "get_paused_stages",
            _get_paused_stages_operation,
            treat_none_as_success=True,
        )
        return result or []
    
    async def cancel_all_paused_stages(self, session_id: str) -> int:
        """
        Cancel all paused stages for a session.
        
        Updates all paused stages to CANCELLED status with proper timestamps
        and duration calculations. Used during session cancellation.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of stages cancelled
        """
        paused_stages = await self.get_paused_stages(session_id)
        if not paused_stages:
            return 0
        
        current_time = now_us()
        for stage in paused_stages:
            stage.status = StageStatus.CANCELLED.value
            stage.error_message = "Cancelled by user"
            # Use paused_at_us if available, otherwise current time
            stage.completed_at_us = stage.paused_at_us or current_time
            # Calculate duration if we have start time
            if stage.started_at_us and stage.completed_at_us:
                duration_us = stage.completed_at_us - stage.started_at_us
                stage.duration_ms = int(duration_us / 1000)
            # Persist the updated stage
            await self.update_stage_execution(stage)
        
        logger.info(f"Cancelled {len(paused_stages)} paused stages for session {session_id}")
        return len(paused_stages)
