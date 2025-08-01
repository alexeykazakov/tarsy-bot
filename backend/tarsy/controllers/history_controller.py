"""
History Controller

FastAPI controller for alert processing history endpoints.
Provides REST API for querying historical data with filtering, pagination,
and chronological timeline reconstruction.
Uses Unix timestamps (microseconds since epoch) throughout for optimal
performance and consistency with the rest of the system.
"""

import csv
import io
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import StreamingResponse

from tarsy.models.api_models import (
    ErrorResponse,
    HealthCheckResponse,
    PaginationInfo,
    SessionDetailResponse,
    SessionsListResponse,
    SessionSummary,
    TimelineEvent,
)
from tarsy.models.history import now_us
from tarsy.services.history_service import HistoryService, get_history_service

router = APIRouter(prefix="/api/v1/history", tags=["history"])

@router.get(
    "/sessions", 
    response_model=SessionsListResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request - invalid parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="List Alert Processing Sessions",
    description="""
    Retrieve a paginated list of alert processing sessions with optional filtering.
    
    **Filtering Support:**
    - Multiple filters use AND logic (all conditions must be met)
    - Combine filters for precise queries (e.g., alert_type + status + time_range)
    
    **Common Use Cases:**
    1. Recent completed alerts: `status=completed&start_date_us=1734476400000000`
    2. Kubernetes alerts: `agent_type=kubernetes&status=completed`
    3. Specific alert types: `alert_type=NamespaceTerminating`
    4. Search error messages: `search=connection refused&status=failed`
    5. Search analysis content: `search=namespace terminating`
    6. Time range analysis: `start_date_us=1734476400000000&end_date_us=1734562799999999`
    
    **Timestamp Format:**
    - All timestamps are Unix timestamps in microseconds since epoch (UTC)
    """
)
async def list_sessions(
    status: Optional[List[str]] = Query(None, description="Filter by session status(es) - supports multiple values"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    alert_type: Optional[str] = Query(None, description="Filter by alert type"),
    search: Optional[str] = Query(None, description="Text search across alert messages, error messages, and analysis results", min_length=3),
    start_date_us: Optional[int] = Query(None, description="Filter sessions started after this timestamp (microseconds since epoch UTC)"),
    end_date_us: Optional[int] = Query(None, description="Filter sessions started before this timestamp (microseconds since epoch UTC)"),
    page: int = Query(1, ge=1, description="Page number for pagination"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page (1-100)"),
    history_service: HistoryService = Depends(get_history_service)
) -> SessionsListResponse:
    """
    List alert processing sessions with filtering and pagination.
    
    Args:
        status: Optional status filter(s) (e.g., ['completed', 'failed'] or ['in_progress'])
        agent_type: Optional agent type filter (e.g., 'kubernetes')
        alert_type: Optional alert type filter (e.g., 'NamespaceTerminating')
        search: Optional text search across alert messages, errors, and analysis (minimum 3 characters)
        start_date_us: Optional start timestamp filter (microseconds since epoch UTC, inclusive)
        end_date_us: Optional end timestamp filter (microseconds since epoch UTC, inclusive)
        page: Page number (starting from 1)
        page_size: Number of items per page (1-100)
        history_service: Injected history service
        
    Returns:
        SessionsListResponse containing paginated session list with filters applied
        
    Raises:
        HTTPException: 400 for invalid parameters, 500 for internal errors
    """
    try:
        # Build filters dictionary
        filters = {}
        if status is not None:
            filters['status'] = status
        if agent_type is not None:
            filters['agent_type'] = agent_type
        if alert_type is not None:
            filters['alert_type'] = alert_type
        if search is not None and search.strip():
            filters['search'] = search.strip()
        if start_date_us is not None:
            filters['start_date_us'] = start_date_us
        if end_date_us is not None:
            filters['end_date_us'] = end_date_us
            
        # Validate timestamp range
        if start_date_us and end_date_us and start_date_us >= end_date_us:
            raise HTTPException(
                status_code=400,
                detail="start_date_us must be before end_date_us"
            )
        
        # Get sessions from history service with pagination
        sessions, total_count = history_service.get_sessions_list(
            filters=filters,
            page=page,
            page_size=page_size
        )
        
        # Convert to response models
        session_summaries = []
        for session in sessions:
            # Calculate duration if completed
            duration_ms = None
            if session.completed_at_us and session.started_at_us:
                duration_ms = int((session.completed_at_us - session.started_at_us) / 1000)
            
            # Get interaction/communication counts (from repository subqueries)
            llm_count = getattr(session, 'llm_interaction_count', 0)
            mcp_count = getattr(session, 'mcp_communication_count', 0)
            
            session_summary = SessionSummary(
                session_id=session.session_id,
                alert_id=session.alert_id,
                agent_type=session.agent_type,
                alert_type=session.alert_type,
                status=session.status,
                started_at_us=session.started_at_us,
                completed_at_us=session.completed_at_us,
                error_message=session.error_message,
                duration_ms=duration_ms,
                llm_interaction_count=llm_count,
                mcp_communication_count=mcp_count
            )
            session_summaries.append(session_summary)
        
        # Calculate pagination info
        total_pages = (total_count + page_size - 1) // page_size
        pagination = PaginationInfo(
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            total_items=total_count
        )
        
        return SessionsListResponse(
            sessions=session_summaries,
            pagination=pagination,
            filters_applied=filters
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve sessions: {str(e)}"
        )

@router.get(
    "/sessions/{session_id}",
    response_model=SessionDetailResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get Session Details with Timeline",
    description="""
    Retrieve detailed information for a specific alert processing session.
    
    **Response includes:**
    - Complete session metadata and processing details
    - Chronological timeline of all LLM interactions and MCP communications
    - Full audit trail with microsecond-precision timing
    
    **Timestamp Format:**
    - All timestamps are Unix timestamps in microseconds since epoch (UTC)
    """
)
async def get_session_detail(
    session_id: str = Path(..., description="Unique session identifier"),
    history_service: HistoryService = Depends(get_history_service)
) -> SessionDetailResponse:
    """
    Get detailed session information with chronological timeline.
    
    Args:
        session_id: Unique session identifier
        history_service: Injected history service
        
    Returns:
        SessionDetailResponse containing complete session details and timeline
        
    Raises:
        HTTPException: 404 if session not found, 500 for internal errors
    """
    try:
        # Get session details from history service
        session_data = history_service.get_session_timeline(session_id)
        
        if not session_data:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        # Extract session information
        session_info = session_data.get('session', {})
        timeline = session_data.get('chronological_timeline', [])
        # No summary in repository response - create empty dict for now
        summary = {}
        
        # Calculate total duration if completed
        duration_ms = None
        started_at_us = session_info.get('started_at_us')
        completed_at_us = session_info.get('completed_at_us')
        if completed_at_us and started_at_us:
            duration_ms = int((completed_at_us - started_at_us) / 1000)
        
        # Convert timeline to response models
        timeline_events = []
        for event in timeline:
            timeline_event = TimelineEvent(
                event_id=event.get('event_id', event.get('interaction_id', event.get('communication_id', 'unknown'))),
                type=event.get('type', 'unknown'),
                timestamp_us=event['timestamp_us'],
                step_description=event.get('step_description', 'No description available'),
                details=event.get('details', {}),
                duration_ms=event.get('duration_ms')
            )
            timeline_events.append(timeline_event)
        
        return SessionDetailResponse(
            session_id=session_info['session_id'],
            alert_id=session_info['alert_id'],
            alert_data=session_info.get('alert_data', {}),
            agent_type=session_info['agent_type'],
            alert_type=session_info.get('alert_type'),
            status=session_info['status'],
            started_at_us=started_at_us,
            completed_at_us=completed_at_us,
            error_message=session_info.get('error_message'),
            final_analysis=session_info.get('final_analysis'),
            duration_ms=duration_ms,
            session_metadata=session_info.get('session_metadata', {}),
            chronological_timeline=timeline_events,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve session details: {str(e)}"
        )

@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="History Service Health Check",
    description="""
    Check the health status of the history service.
    
    **Status Values:**
    - `healthy`: Service is operational and database is accessible
    - `unhealthy`: Service has issues (e.g., database connection failed)
    - `disabled`: Service is disabled via configuration (HISTORY_ENABLED=false)
    
    **Timestamp Format:**
    - All timestamps are Unix timestamps in microseconds since epoch (UTC)
    """
)
async def health_check(
    history_service: HistoryService = Depends(get_history_service)
) -> HealthCheckResponse:
    """
    Perform history service health check.
    
    Args:
        history_service: Injected history service
        
    Returns:
        HealthCheckResponse containing service health status and details
    """
    try:
        # Check if history service is enabled
        if not history_service.enabled:
            return HealthCheckResponse(
                service="alert_processing_history",
                status="disabled",
                timestamp_us=now_us(),
                details={
                    "message": "History service is disabled via configuration",
                    "history_enabled": False
                }
            )
        
        # Test database connectivity
        test_result = history_service.test_database_connection()
        
        if test_result:
            return HealthCheckResponse(
                service="alert_processing_history",
                status="healthy",
                timestamp_us=now_us(),
                details={
                    "database_connection": "ok",
                    "history_enabled": True,
                    "database_url": history_service.settings.history_database_url.split('/')[-1]  # Just the DB name for security
                }
            )
        else:
            return HealthCheckResponse(
                service="alert_processing_history",
                status="unhealthy",
                timestamp_us=now_us(),
                details={
                    "database_connection": "failed",
                    "history_enabled": True,
                    "error": "Database connection test failed"
                }
            )
            
    except Exception as e:
        return HealthCheckResponse(
            service="alert_processing_history",
            status="unhealthy",
            timestamp_us=now_us(),
            details={
                "error": str(e),
                "message": "Health check failed with exception"
            }
        )

# Dashboard-specific endpoints for EP-0004 implementation

@router.get(
    "/metrics",
    summary="Dashboard Metrics",
    description="Get dashboard overview metrics for active and completed sessions"
)
async def get_dashboard_metrics(
    history_service: HistoryService = Depends(get_history_service)
):
    """Get dashboard metrics including session counts and status distribution."""
    try:
        return history_service.get_dashboard_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get(
    "/active-sessions",
    summary="Active Sessions", 
    description="Get currently active/processing sessions"
)
async def get_active_sessions(
    history_service: HistoryService = Depends(get_history_service)
):
    """Get list of currently active sessions."""
    try:
        active_sessions = history_service.get_active_sessions()
        # Convert to the format expected by the frontend
        return [
            {
                "session_id": session.session_id,
                "alert_id": session.alert_id,
                "agent_type": session.agent_type,
                "alert_type": session.alert_type,
                "status": session.status,
                "started_at_us": session.started_at_us,
                "completed_at_us": session.completed_at_us,
                "error_message": session.error_message,
                "duration_seconds": (
                    (session.completed_at_us - session.started_at_us) / 1000000
                    if session.completed_at_us else 
                    (now_us() - session.started_at_us) / 1000000
                )
            }
            for session in active_sessions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active sessions: {str(e)}")

@router.get(
    "/filter-options",
    summary="Filter Options",
    description="Get available filter options for dashboard filtering"
)
async def get_filter_options(
    history_service: HistoryService = Depends(get_history_service)
):
    """Get available filter options for the dashboard."""
    try:
        return history_service.get_filter_options()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get filter options: {str(e)}")

@router.get(
    "/sessions/{session_id}/export",
    summary="Export Session Data",
    description="Export session data with timeline in JSON or CSV format"
)
async def export_session_data(
    session_id: str = Path(..., description="Session ID to export"),
    format: str = Query("json", regex="^(json|csv)$", description="Export format: json or csv"),
    history_service: HistoryService = Depends(get_history_service)
):
    """Export comprehensive session data including timeline."""
    try:
        export_result = history_service.export_session_data(session_id, format)
        
        if export_result.get("error"):
            if "not found" in export_result["error"].lower():
                raise HTTPException(status_code=404, detail=export_result["error"])
            else:
                raise HTTPException(status_code=500, detail=export_result["error"])
        
        export_data = export_result["data"]
        
        if format == "csv":
            # Generate CSV format
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write session header
            writer.writerow(["Session Information"])
            writer.writerow(["Field", "Value"])
            writer.writerow(["Session ID", export_data["session"]["session_id"]])
            writer.writerow(["Alert ID", export_data["session"]["alert_id"]])
            writer.writerow(["Agent Type", export_data["session"]["agent_type"]])
            writer.writerow(["Alert Type", export_data["session"]["alert_type"]])
            writer.writerow(["Status", export_data["session"]["status"]])
            writer.writerow(["Started At (us)", export_data["session"]["started_at_us"]])
            writer.writerow(["Completed At (us)", export_data["session"]["completed_at_us"] or "N/A"])
            writer.writerow(["Error Message", export_data["session"]["error_message"] or "None"])
            writer.writerow([])
            
            # Write timeline interactions
            writer.writerow(["Timeline Interactions"])
            writer.writerow(["Timestamp (us)", "Type", "Description", "Duration (ms)", "Success"])
            
            for interaction in export_data["timeline"].get("interactions", []):
                writer.writerow([
                    interaction.get("timestamp_us", ""),
                    interaction.get("type", ""),
                    interaction.get("description", "")[:100] + "..." if len(interaction.get("description", "")) > 100 else interaction.get("description", ""),
                    interaction.get("duration_ms", ""),
                    interaction.get("success", "")
                ])
            
            csv_content = output.getvalue()
            output.close()
            
            return StreamingResponse(
                io.BytesIO(csv_content.encode('utf-8')),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=session_{session_id}.csv"}
            )
        else:
            # Return JSON format
            return export_data
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export session data: {str(e)}")

@router.get(
    "/search",
    summary="Search Sessions",
    description="Search sessions by alert content, error messages, or metadata"
)
async def search_sessions(
    q: str = Query(..., description="Search query string"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    history_service: HistoryService = Depends(get_history_service)
):
    """Search sessions by various fields."""
    try:
        results = history_service.search_sessions(q, limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search sessions: {str(e)}")

# Note: Exception handlers should be registered at the app level in main.py 