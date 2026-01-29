"""
Base Infrastructure for Session Data Service.

Provides shared infrastructure components used by all session data operations
including database management, retry logic, and health tracking.
"""

import asyncio
import logging
import random
import time
from contextlib import contextmanager
from typing import Optional

from tarsy.config.settings import get_settings
from tarsy.repositories.base_repository import DatabaseManager
from tarsy.repositories.history_repository import HistoryRepository

logger = logging.getLogger(__name__)


class BaseSessionDataInfra:
    """
    Base infrastructure for session data operations.
    
    Provides shared components for database access, retry logic with exponential
    backoff, and health status tracking. All specialized operation classes compose
    this base to access these shared facilities.
    """
    
    def __init__(self):
        """Initialize base infrastructure with configuration."""
        self.settings = get_settings()
        self.db_manager: Optional[DatabaseManager] = None
        self._initialization_attempted = False
        self._is_healthy = False
        
        # Retry configuration for database operations
        self.max_retries = 3
        self.base_delay = 0.1  # 100ms base delay
        self.max_delay = 2.0   # 2 second max delay
    
    def initialize(self) -> bool:
        """
        Initialize database connection and schema.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialization_attempted:
            return self._is_healthy
            
        self._initialization_attempted = True
        
        try:
            self.db_manager = DatabaseManager(self.settings.database_url)
            self.db_manager.initialize()
            self.db_manager.create_tables()
            self._is_healthy = True
            logger.info("Session data service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize session data service: {str(e)}")
            logger.info("Session data service will operate in degraded mode (logging only)")
            self._is_healthy = False
            return False
    
    @contextmanager
    def get_repository(self):
        """
        Context manager for getting repository with error handling.
        
        Yields:
            HistoryRepository instance or None if unavailable
        """
        if not self._is_healthy:
            yield None
            return
            
        session = None
        repository = None
        try:
            if not self.db_manager:
                yield None
                return
                
            session = self.db_manager.get_session()
            repository = HistoryRepository(session)
            yield repository
            
        except Exception as e:
            logger.error(f"Repository error: {str(e)}")
            if session:
                try:
                    session.rollback()
                except Exception:
                    pass
            
        finally:
            if session:
                try:
                    session.close()
                except Exception as e:
                    logger.error(f"Error closing database session: {str(e)}")
    
    def _retry_database_operation(
        self,
        operation_name: str,
        operation_func,
        *,
        treat_none_as_success: bool = False,
    ):
        """
        Retry database operations with exponential backoff for transient failures.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to retry
            treat_none_as_success: Whether None result is acceptable
        
        Returns:
            Result of the operation, or None if all retries failed
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = operation_func()
                if result is not None:
                    return result
                if treat_none_as_success:
                    # None is an acceptable outcome (e.g., entity not found)
                    return None
                # If result is None, log and continue to retry
                logger.warning(
                    f"Database operation '{operation_name}' returned None on attempt {attempt + 1}"
                )
                
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Check if this is a retryable database error
                is_retryable = any(keyword in error_msg for keyword in [
                    'database is locked',
                    'database disk image is malformed', 
                    'sqlite3.operationalerror',
                    'connection timeout',
                    'database table is locked',
                    'connection pool',
                    'connection closed'
                ])
                
                # For session creation, check if it might have succeeded despite the error
                if operation_name == "create_session" and attempt > 0:
                    # Don't retry if we might have created a duplicate
                    logger.warning(f"Not retrying session creation after database error to prevent duplicates: {str(e)}")
                    return None
                
                if not is_retryable or attempt == self.max_retries:
                    logger.error(f"Database operation '{operation_name}' failed after {attempt + 1} attempts: {str(e)}")
                    return None
                
                # Exponential backoff with jitter
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                jitter = random.uniform(0, delay * 0.1)  # Add up to 10% jitter
                total_delay = delay + jitter
                
                logger.warning(f"Database operation '{operation_name}' failed on attempt {attempt + 1}, retrying in {total_delay:.2f}s: {str(e)}")
                time.sleep(total_delay)
        
        logger.error(f"Database operation '{operation_name}' failed after all retries. Last error: {str(last_exception)}")
        return None
    
    async def _retry_database_operation_async(
        self,
        operation_name: str,
        operation_func,
        *,
        treat_none_as_success: bool = False,
    ):
        """
        Async version of retry database operations with exponential backoff for transient failures.
        
        Runs synchronous database operations in a thread pool to avoid blocking the event loop.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to retry (executed in thread pool)
            treat_none_as_success: Whether None result is acceptable
        
        Returns:
            Result of the operation, or None if all retries failed
        """
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.to_thread(operation_func)
                if result is not None:
                    return result
                if treat_none_as_success:
                    return None
                logger.warning(f"Database operation '{operation_name}' returned None on attempt {attempt + 1}")
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                is_retryable = any(k in error_msg for k in [
                    'database is locked', 'database disk image is malformed', 'sqlite3.operationalerror',
                    'connection timeout', 'database table is locked', 'connection pool', 'connection closed'
                ])
                if operation_name == "create_session" and attempt > 0:
                    logger.warning("Not retrying session creation after database error to prevent duplicates: %s", str(e))
                    return None
                if not is_retryable or attempt == self.max_retries:
                    logger.error("Database operation '%s' failed after %d attempts: %s", operation_name, attempt + 1, str(e))
                    return None
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                jitter = random.uniform(0, delay * 0.1)
                await asyncio.sleep(delay + jitter)
        logger.error("Database operation '%s' failed after all retries. Last error: %s", operation_name, str(last_exception))
        return None
