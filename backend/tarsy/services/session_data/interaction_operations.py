"""
Interaction Operations for Session Data Service.

Handles LLM and MCP interaction logging for audit trail purposes.
"""

import logging

from tarsy.models.unified_interactions import LLMInteraction, MCPInteraction
from tarsy.services.session_data.base_infrastructure import BaseSessionDataInfra

logger = logging.getLogger(__name__)


class InteractionOperations:
    """
    Interaction logging operations.
    
    Manages storage of LLM and MCP interactions for audit trail and timeline
    reconstruction purposes.
    """
    
    def __init__(self, infra: BaseSessionDataInfra):
        """
        Initialize interaction operations with shared infrastructure.
        
        Args:
            infra: Shared infrastructure instance
        """
        self._infra = infra
    
    def store_llm_interaction(self, interaction: LLMInteraction) -> bool:
        """
        Store an LLM interaction to the database using unified model.
        
        Args:
            interaction: LLMInteraction instance with all interaction details
            
        Returns:
            True if logged successfully, False otherwise
        """
        if not interaction.session_id:
            return False
            
        def _store_llm_operation():
            with self._infra.get_repository() as repo:
                if not repo:
                    raise RuntimeError("Repository unavailable - cannot store LLM interaction")
                # Step description is handled at the timeline event level, not interaction level
                repo.create_llm_interaction(interaction)
                logger.debug(f"Stored LLM interaction for session {interaction.session_id}")
                return True

        result = self._infra._retry_database_operation("store_llm_interaction", _store_llm_operation)
        return bool(result)
    
    def store_mcp_interaction(self, interaction: MCPInteraction) -> bool:
        """
        Store an MCP interaction to the database using unified model.
        
        Args:
            interaction: MCPInteraction instance with all interaction details
            
        Returns:
            True if logged successfully, False otherwise
        """
        if not interaction.session_id:
            return False
            
        def _store_mcp_operation():
            with self._infra.get_repository() as repo:
                if not repo:
                    raise RuntimeError("Repository unavailable - cannot store MCP interaction")
                # Set step description if not already set
                if not interaction.step_description:
                    interaction.step_description = interaction.get_step_description()
                repo.create_mcp_communication(interaction)
                logger.debug(f"Stored MCP interaction for session {interaction.session_id}")
                return True

        result = self._infra._retry_database_operation("store_mcp_interaction", _store_mcp_operation)
        return bool(result)
