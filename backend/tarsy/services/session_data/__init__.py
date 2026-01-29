"""
Session Data Service for Alert Processing Audit Trail.

Provides centralized management of alert processing session data including session
lifecycle, LLM interaction tracking, MCP communication logging, and timeline
reconstruction with graceful degradation when database is unavailable.

Renamed from HistoryService to SessionDataService to accurately reflect its role
in managing active session data (create, update, query, track), not just historical queries.
"""

from typing import Optional

from .session_data_service import SessionDataService

# Global singleton instance
_session_data_service: Optional[SessionDataService] = None


def get_session_data_service() -> SessionDataService:
    """
    Get the global session data service instance.
    
    Returns:
        SessionDataService instance (initialized on first access)
    """
    global _session_data_service
    if _session_data_service is None:
        _session_data_service = SessionDataService()
        _session_data_service.initialize()
    return _session_data_service


# Public API
__all__ = ['SessionDataService', 'get_session_data_service']
