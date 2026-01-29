"""
Query Operations for Session Data Service.

Handles session querying, filtering, and retrieval operations for dashboard
and API endpoints.
"""

import logging
from typing import Any, Dict, List, Optional

from tarsy.models.db_models import AlertSession
from tarsy.models.history_models import DetailedSession, FilterOptions, PaginatedSessions
from tarsy.services.session_data.base_infrastructure import BaseSessionDataInfra

logger = logging.getLogger(__name__)


class QueryOperations:
    """
    Session query operations.
    
    Manages session list queries, filtering, pagination, and detailed session
    retrieval for dashboard and API consumption.
    """
    
    def __init__(self, infra: BaseSessionDataInfra):
        """
        Initialize query operations with shared infrastructure.
        
        Args:
            infra: Shared infrastructure instance
        """
        self._infra = infra
    
    def get_sessions_list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None
    ) -> Optional[PaginatedSessions]:
        """
        Retrieve alert sessions with filtering and pagination - returns PaginatedSessions model directly for controllers.
        
        Args:
            filters: Dictionary of filters (status, agent_type, alert_type, start_date_us, end_date_us)
            page: Page number for pagination
            page_size: Number of results per page
            sort_by: Optional field to sort by
            sort_order: Optional sort order ('asc' or 'desc')
            
        Returns:
            PaginatedSessions model or None if unavailable
            
        Raises:
            RuntimeError: If repository is unavailable
        """
        with self._infra.get_repository() as repo:
            if not repo:
                raise RuntimeError("Repository unavailable - cannot retrieve sessions list")
            
            # Extract filters or use defaults
            filters = filters or {}
            
            # Use regular method that returns PaginatedSessions model directly
            paginated_sessions = repo.get_alert_sessions(
                status=filters.get('status'),
                agent_type=filters.get('agent_type'),
                alert_type=filters.get('alert_type'),
                search=filters.get('search'),
                start_date_us=filters.get('start_date_us'),
                end_date_us=filters.get('end_date_us'),
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                sort_order=sort_order
            )
            
            # Add the applied filters to the model
            if paginated_sessions and filters:
                paginated_sessions.filters_applied = filters
            
            return paginated_sessions

    def test_database_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            True if database connection is working, False otherwise
        """
        try:
            with self._infra.get_repository() as repo:
                if not repo:
                    raise RuntimeError("Repository unavailable - cannot check health")
                
                # Try to perform a simple database operation
                # This will test both connection and basic functionality
                repo.get_alert_sessions(page=1, page_size=1)
                # Result is now PaginatedSessions model or None
                return True
                
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False

    def get_session_details(self, session_id: str) -> Optional[DetailedSession]:
        """
        Get complete session details including timeline, stages, and all interactions.
        
        Args:
            session_id: The session identifier
            
        Returns:
            DetailedSession model with complete session data or None if not found
        """
        try:
            with self._infra.get_repository() as repo:
                if not repo:
                    raise RuntimeError("Repository unavailable - cannot retrieve session details")
                
                detailed_session = repo.get_session_details(session_id)
                return detailed_session
                
        except Exception as e:
            logger.error(f"Failed to get session details for {session_id}: {str(e)}")
            return None
    
    def get_active_sessions(self) -> List[AlertSession]:
        """
        Get all currently active sessions.
        
        Returns:
            List of active AlertSession instances
            
        Raises:
            RuntimeError: If repository is unavailable
        """
        with self._infra.get_repository() as repo:
            if not repo:
                raise RuntimeError("Repository unavailable - cannot retrieve active sessions")
            
            return repo.get_active_sessions()

    def get_filter_options(self) -> FilterOptions:
        """
        Get available filter options for the dashboard - returns FilterOptions model for controllers.
        
        Returns:
            FilterOptions model
            
        Raises:
            RuntimeError: If repository is unavailable
        """
        with self._infra.get_repository() as repo:
            if not repo:
                raise RuntimeError("Repository unavailable - cannot retrieve filter options")
            
            return repo.get_filter_options()
