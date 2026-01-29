"""
Queue Operations for Session Data Service.

Handles session queue management for global alert processing queue including
session counting and atomic claiming operations.
"""

import logging
from typing import Optional

from tarsy.models.db_models import AlertSession
from tarsy.services.session_data.base_infrastructure import BaseSessionDataInfra

logger = logging.getLogger(__name__)


class QueueOperations:
    """
    Queue management operations.
    
    Manages global session queue operations including session counting by status
    and atomic claiming of pending sessions for processing.
    """
    
    def __init__(self, infra: BaseSessionDataInfra):
        """
        Initialize queue operations with shared infrastructure.
        
        Args:
            infra: Shared infrastructure instance
        """
        self._infra = infra
    
    def count_sessions_by_status(self, status: str) -> int:
        """
        Count sessions with given status across all pods.
        
        Args:
            status: Session status to count (e.g., AlertSessionStatus.IN_PROGRESS.value)
            
        Returns:
            Count of sessions with the given status
        """
        with self._infra.get_repository() as repo:
            if not repo:
                return 0
            return repo.count_sessions_by_status(status)
    
    def count_pending_sessions(self) -> int:
        """
        Count sessions in PENDING state (for queue size check).
        
        Returns:
            Count of pending sessions
        """
        with self._infra.get_repository() as repo:
            if not repo:
                return 0
            return repo.count_pending_sessions()
    
    def claim_next_pending_session(self, pod_id: str) -> Optional[AlertSession]:
        """
        Atomically claim next PENDING session for this pod.
        
        Args:
            pod_id: Pod identifier claiming the session
            
        Returns:
            Claimed AlertSession if available, None otherwise
        """
        with self._infra.get_repository() as repo:
            if not repo:
                return None
            return repo.claim_next_pending_session(pod_id)
