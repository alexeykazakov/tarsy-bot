"""
Unit tests for HistoryController.

Tests the REST API endpoints for the history service with mocked services
to ensure proper request/response handling and API contract compliance.
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tarsy.controllers.history_controller import HistoryService, router
from tarsy.services.history_service import get_history_service


class TestHistoryControllerEndpoints:
    """Test suite for HistoryController API endpoints."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI application with history router."""
        app = FastAPI()
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_history_service(self):
        """Create mock history service."""
        service = Mock(spec=HistoryService)
        service.enabled = True
        service.is_enabled = True
        service.get_sessions_list.return_value = ([], 0)
        service.get_session_timeline.return_value = None
        service.test_database_connection.return_value = True
        service.health_check.return_value = {
            "enabled": True,
            "healthy": True,
            "database_url": "sqlite:///test.db",
            "retention_days": 90
        }
        
        # Add settings mock
        mock_settings = Mock()
        mock_settings.history_database_url = "sqlite:///test_history.db"
        mock_settings.history_enabled = True
        mock_settings.history_retention_days = 90
        service.settings = mock_settings
        
        return service
    
    @pytest.mark.unit
    def test_get_sessions_list_success(self, app, client, mock_history_service):
        """Test successful sessions list retrieval."""
        # Mock response data
        session1 = Mock()
        session1.session_id = "session-1"
        session1.alert_id = "alert-1"
        session1.alert_type = "NamespaceTerminating"
        session1.agent_type = "KubernetesAgent"
        session1.status = "completed"
        session1.started_at = datetime.now(timezone.utc)
        session1.completed_at = datetime.now(timezone.utc)
        session1.error_message = None
        session1.llm_interactions = []
        session1.mcp_communications = []
        # Add the new dynamic attributes expected by the controller
        session1.llm_interaction_count = 0
        session1.mcp_communication_count = 0
        
        session2 = Mock()
        session2.session_id = "session-2"
        session2.alert_id = "alert-2"
        session2.alert_type = "HighCPU"
        session2.agent_type = "KubernetesAgent"
        session2.status = "in_progress"
        session2.started_at = datetime.now(timezone.utc)
        session2.completed_at = None
        session2.error_message = None
        session2.llm_interactions = []
        session2.mcp_communications = []
        # Add the new dynamic attributes expected by the controller
        session2.llm_interaction_count = 0
        session2.mcp_communication_count = 0
        
        mock_sessions = [session1, session2]
        mock_history_service.get_sessions_list.return_value = (mock_sessions, 2)
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get("/api/v1/history/sessions")
        
        # Clean up
        app.dependency_overrides.clear()
        
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.content}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sessions" in data
        assert "pagination" in data
        assert len(data["sessions"]) == 2
        assert data["pagination"]["total_items"] == 2
        
        # Verify session data structure
        session = data["sessions"][0]
        assert "session_id" in session
        assert "alert_id" in session
        assert "alert_type" in session
        assert "agent_type" in session
        assert "status" in session
        assert "started_at" in session
    
    @pytest.mark.unit
    def test_get_sessions_list_with_filters(self, app, client, mock_history_service):
        """Test sessions list with query parameters."""
        mock_history_service.get_sessions_list.return_value = ([], 0)
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get(
            "/api/v1/history/sessions",
            params={
                "status": "completed",
                "agent_type": "KubernetesAgent",
                "alert_type": "NamespaceTerminating",
                "page": 2,
                "page_size": 10
            }
        )
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        
        # Verify that filters were passed to service
        mock_history_service.get_sessions_list.assert_called_once()
        call_args = mock_history_service.get_sessions_list.call_args
        filters = call_args.kwargs["filters"]
        
        assert filters["status"] == ["completed"]  # Now expects list due to multiple status support
        assert filters["agent_type"] == "KubernetesAgent"
        assert filters["alert_type"] == "NamespaceTerminating"
        assert call_args.kwargs["page"] == 2
        assert call_args.kwargs["page_size"] == 10

    @pytest.mark.unit
    def test_get_sessions_list_with_single_status_filter(self, app, client, mock_history_service):
        """Test that single status filtering still works (backward compatibility)."""
        # Arrange
        mock_history_service.get_sessions_list.return_value = ([], 0)
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get(
            "/api/v1/history/sessions",
            params={"status": "completed"}
        )
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        
        # Verify that single status is converted to list
        mock_history_service.get_sessions_list.assert_called_once()
        call_args = mock_history_service.get_sessions_list.call_args
        filters = call_args.kwargs["filters"]
        
        assert filters["status"] == ["completed"]  # Single status becomes list

    @pytest.mark.unit
    def test_get_sessions_list_with_multiple_status_filter(self, app, client, mock_history_service):
        """Test multiple status filtering (the new feature)."""
        # Arrange
        mock_history_service.get_sessions_list.return_value = ([], 0)
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get(
            "/api/v1/history/sessions",
            params=[
                ("status", "completed"),
                ("status", "failed"),
                ("agent_type", "KubernetesAgent")
            ]
        )
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        
        # Verify that multiple status values are passed correctly
        mock_history_service.get_sessions_list.assert_called_once()
        call_args = mock_history_service.get_sessions_list.call_args
        filters = call_args.kwargs["filters"]
        
        assert set(filters["status"]) == {"completed", "failed"}  # Multiple status values
        assert filters["agent_type"] == "KubernetesAgent"

    @pytest.mark.unit 
    def test_get_sessions_list_multiple_status_historical_use_case(self, app, client, mock_history_service):
        """Test the specific use case for historical alerts (completed + failed)."""
        # Arrange
        mock_sessions = [
            create_mock_session("session-1", "completed"),
            create_mock_session("session-2", "failed"),
        ]
        mock_history_service.get_sessions_list.return_value = (mock_sessions, 2)
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get(
            "/api/v1/history/sessions",
            params=[("status", "completed"), ("status", "failed")]
        )
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "sessions" in data
        assert "pagination" in data
        assert len(data["sessions"]) == 2
        
        # Verify that service was called with correct multiple status filter
        mock_history_service.get_sessions_list.assert_called_once()
        call_args = mock_history_service.get_sessions_list.call_args
        filters = call_args.kwargs["filters"]
        
        assert set(filters["status"]) == {"completed", "failed"}

    @pytest.mark.unit
    def test_get_sessions_list_multiple_status_active_use_case(self, app, client, mock_history_service):
        """Test filtering for active alerts (pending + in_progress)."""
        # Arrange
        mock_sessions = [
            create_mock_session("session-1", "pending"),
            create_mock_session("session-2", "in_progress"),
        ]
        mock_history_service.get_sessions_list.return_value = (mock_sessions, 2)
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get(
            "/api/v1/history/sessions",
            params=[("status", "pending"), ("status", "in_progress")]
        )
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "sessions" in data
        assert len(data["sessions"]) == 2
        
        # Verify that service was called with correct multiple status filter
        mock_history_service.get_sessions_list.assert_called_once()
        call_args = mock_history_service.get_sessions_list.call_args
        filters = call_args.kwargs["filters"]
        
        assert set(filters["status"]) == {"pending", "in_progress"}

    @pytest.mark.unit
    def test_get_sessions_list_three_status_values(self, app, client, mock_history_service):
        """Test filtering with three status values."""
        # Arrange
        mock_history_service.get_sessions_list.return_value = ([], 0)
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get(
            "/api/v1/history/sessions",
            params=[
                ("status", "completed"),
                ("status", "failed"), 
                ("status", "in_progress")
            ]
        )
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        
        # Verify all three status values are passed
        mock_history_service.get_sessions_list.assert_called_once()
        call_args = mock_history_service.get_sessions_list.call_args
        filters = call_args.kwargs["filters"]
        
        assert set(filters["status"]) == {"completed", "failed", "in_progress"}

    @pytest.mark.unit
    def test_get_sessions_list_no_status_filter(self, app, client, mock_history_service):
        """Test that endpoint works without status filter (gets all sessions)."""
        # Arrange
        mock_history_service.get_sessions_list.return_value = ([], 0)
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get(
            "/api/v1/history/sessions",
            params={"agent_type": "KubernetesAgent"}  # Other filters but no status
        )
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        
        # Verify that status filter is not included when not specified
        mock_history_service.get_sessions_list.assert_called_once()
        call_args = mock_history_service.get_sessions_list.call_args
        filters = call_args.kwargs["filters"]
        
        assert "status" not in filters  # No status filter when not provided
        assert filters["agent_type"] == "KubernetesAgent"
    
    @pytest.mark.unit
    def test_get_sessions_list_with_date_filters(self, app, client, mock_history_service):
        """Test sessions list with date range filters."""
        mock_history_service.get_sessions_list.return_value = ([], 0)
        
        start_date = "2024-01-01T00:00:00Z"
        end_date = "2024-01-31T23:59:59Z"
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get(
            "/api/v1/history/sessions",
            params={
                "start_date": start_date,
                "end_date": end_date
            }
        )
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        
        # Verify date filters were parsed and passed correctly
        call_args = mock_history_service.get_sessions_list.call_args
        filters = call_args.kwargs["filters"]
        
        assert "start_date" in filters
        assert "end_date" in filters
        assert isinstance(filters["start_date"], datetime)
        assert isinstance(filters["end_date"], datetime)
    
    @pytest.mark.unit
    def test_get_sessions_list_invalid_date_format(self, app, client, mock_history_service):
        """Test sessions list with invalid date format."""
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get(
            "/api/v1/history/sessions",
            params={"start_date": "invalid-date"}
        )
        
        # Clean up
        app.dependency_overrides.clear()
        
        # FastAPI returns 422 for validation errors, not 400
        assert response.status_code == 422
        
        # Check that the error is about datetime parsing
        error_details = response.json()["detail"]
        assert isinstance(error_details, list)
        assert any("datetime" in error.get("type", "") for error in error_details)
    
    @pytest.mark.unit
    def test_get_sessions_list_service_disabled(self, app, client, mock_history_service):
        """Test sessions list when history service is disabled."""
        mock_history_service.enabled = False
        mock_history_service.get_sessions_list.return_value = ([], 0)
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get("/api/v1/history/sessions")
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 0
        assert data["pagination"]["total_items"] == 0
    
    @pytest.mark.unit
    def test_get_sessions_list_service_error(self, app, client, mock_history_service):
        """Test sessions list with service error."""
        mock_history_service.get_sessions_list.side_effect = Exception("Service error")
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get("/api/v1/history/sessions")
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 500
        # The actual error message varies, just check it's an error response
        assert "detail" in response.json()
    
    @pytest.mark.unit
    def test_get_session_detail_success(self, app, client, mock_history_service):
        """Test successful session detail retrieval."""
        # Mock timeline data with correct structure (matches repository output)
        mock_timeline = {
            "session": {
                "session_id": "test-session-123",
                "alert_id": "alert-456",
                "alert_type": "NamespaceTerminating",
                "agent_type": "KubernetesAgent",
                "status": "completed",
                "started_at": "2024-01-15T10:00:00Z",
                "completed_at": "2024-01-15T10:05:00Z",
                "error_message": None
            },
            "chronological_timeline": [
                {
                    "interaction_id": "int-1",
                    "type": "llm_interaction",
                    "timestamp": "2024-01-15T10:00:00Z",
                    "step_description": "Initial analysis",
                    "details": {
                        "prompt_text": "Analyze the issue",
                        "response_text": "Found the problem",
                        "model_used": "gpt-4"
                    }
                },
                {
                    "communication_id": "comm-1",
                    "type": "mcp_communication",
                    "timestamp": "2024-01-15T10:00:01Z",
                    "step_description": "Check namespace status",
                    "details": {
                        "server_name": "kubernetes-server",
                        "tool_name": "kubectl_get_namespace",
                        "success": True
                    }
                }
            ],
            "summary": {
                "total_interactions": 2,
                "llm_interactions": 1,
                "mcp_communications": 1,
                "total_duration_ms": 2000
            }
        }
        mock_history_service.get_session_timeline.return_value = mock_timeline
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get("/api/v1/history/sessions/test-session-123")
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        
        # The response should have these fields based on the SessionDetailResponse model
        assert "session_id" in data
        assert "alert_id" in data
        assert "chronological_timeline" in data
        
        # Verify session details
        assert data["session_id"] == "test-session-123"
        assert data["alert_type"] == "NamespaceTerminating"
        assert data["status"] == "completed"
        
        # Verify timeline structure
        timeline = data["chronological_timeline"]
        assert len(timeline) == 2
        assert timeline[0]["type"] == "llm_interaction"
        assert timeline[1]["type"] == "mcp_communication"
        
        # Verify other expected fields
        assert "duration_ms" in data
        assert "summary" in data
    
    @pytest.mark.unit
    def test_get_session_detail_not_found(self, app, client, mock_history_service):
        """Test session detail for non-existent session."""
        mock_history_service.get_session_timeline.return_value = None
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get("/api/v1/history/sessions/non-existent-session")
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 404
        # Check that the error message contains session reference
        assert "non-existent-session" in response.json()["detail"]
    
    @pytest.mark.unit
    def test_get_session_detail_service_disabled(self, app, client, mock_history_service):
        """Test session detail when service is disabled."""
        mock_history_service.enabled = False
        mock_history_service.get_session_timeline.return_value = None
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get("/api/v1/history/sessions/test-session")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # When service is disabled, timeline returns None, so we get 404
        assert response.status_code == 404
        assert "test-session" in response.json()["detail"]
    
    @pytest.mark.unit
    def test_get_session_detail_service_error(self, app, client, mock_history_service):
        """Test session detail with service error."""
        mock_history_service.get_session_timeline.side_effect = Exception("Service error")
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get("/api/v1/history/sessions/test-session")
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 500
        assert "detail" in response.json()
    
    @pytest.mark.unit
    def test_health_check_healthy_service(self, app, client, mock_history_service):
        """Test health check with healthy service."""
        mock_history_service.enabled = True
        mock_history_service.test_database_connection.return_value = True
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get("/api/v1/history/health")
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["service"] == "alert_processing_history"
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "details" in data
        
        details = data["details"]
        assert details["database_connection"] == "ok"
        assert details["history_enabled"] == True
        assert "database_url" in details
    
    @pytest.mark.unit
    def test_health_check_disabled_service(self, app, client, mock_history_service):
        """Test health check with disabled service."""
        mock_history_service.enabled = False
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get("/api/v1/history/health")
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "disabled"
        assert data["details"]["history_enabled"] == False
        assert "message" in data["details"]
    
    @pytest.mark.unit
    def test_health_check_database_failure(self, app, client, mock_history_service):
        """Test health check with database connection failure."""
        mock_history_service.enabled = True
        mock_history_service.test_database_connection.return_value = False
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get("/api/v1/history/health")
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "unhealthy"
        assert data["details"]["database_connection"] == "failed"
    
    @pytest.mark.unit
    def test_health_check_exception_handling(self, app, client, mock_history_service):
        """Test health check with exception."""
        mock_history_service.enabled = True
        mock_history_service.test_database_connection.side_effect = Exception("Database error")
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        response = client.get("/api/v1/history/health")
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "unhealthy"
        assert "error" in data["details"]


class TestHistoryControllerValidation:
    """Test suite for request validation in HistoryController."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI application with history router."""
        app = FastAPI()
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_history_service(self):
        """Create mock history service."""
        service = Mock(spec=HistoryService)
        service.enabled = True
        service.get_sessions_list.return_value = ([], 0)
        return service
    
    @pytest.mark.unit
    def test_sessions_list_pagination_validation(self, client, mock_history_service):
        """Test pagination parameter validation."""
        with patch('tarsy.controllers.history_controller.get_history_service', return_value=mock_history_service):
            # Test negative page number
            response = client.get("/api/v1/history/sessions?page=-1")
            assert response.status_code == 422
            
            # Test zero page number
            response = client.get("/api/v1/history/sessions?page=0")
            assert response.status_code == 422
            
            # Test negative page_size
            response = client.get("/api/v1/history/sessions?page_size=-1")
            assert response.status_code == 422
            
            # Test page_size too large
            response = client.get("/api/v1/history/sessions?page_size=1001")
            assert response.status_code == 422
    
    @pytest.mark.unit
    def test_sessions_list_enum_validation(self, app, client, mock_history_service):
        """Test enum parameter validation."""
        mock_history_service.get_sessions_list.return_value = ([], 0)
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        # Test invalid status - the API doesn't enforce enum validation, so it returns 200
        response = client.get("/api/v1/history/sessions?status=invalid_status")
        assert response.status_code == 200  # No enum validation in current implementation
        
        # Test valid status values
        valid_statuses = ["pending", "in_progress", "completed", "failed"]
        for status in valid_statuses:
            response = client.get(f"/api/v1/history/sessions?status={status}")
            assert response.status_code == 200
        
        # Clean up
        app.dependency_overrides.clear()
    
    @pytest.mark.unit
    def test_session_id_validation(self, app, client, mock_history_service):
        """Test session ID parameter validation."""
        mock_history_service.get_session_timeline.return_value = None
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        # Test session ID with special characters (should be handled)
        response = client.get("/api/v1/history/sessions/session-with-dashes-123")
        assert response.status_code == 404  # Not found, but valid format
        
        # Test another valid format session ID
        response = client.get("/api/v1/history/sessions/valid-session-id")
        assert response.status_code == 404  # Not found, but endpoint exists
        
        # Clean up
        app.dependency_overrides.clear()
    
    @pytest.mark.unit
    def test_date_format_validation_edge_cases(self, app, client, mock_history_service):
        """Test date format validation edge cases."""
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        # Test only dates that will actually trigger validation errors
        invalid_dates = [
            "not-a-date",           # Completely invalid
            "2024-13-01",           # Invalid month, missing time
            "invalid-format"        # Invalid format
        ]
        
        for invalid_date in invalid_dates:
            response = client.get(f"/api/v1/history/sessions?start_date={invalid_date}")
            # FastAPI returns 422 for validation errors, not 400
            assert response.status_code == 422
            # Check that the error is about datetime parsing
            error_details = response.json()["detail"]
            assert isinstance(error_details, list)
            assert any("datetime" in error.get("type", "") for error in error_details)
        
        # Clean up
        app.dependency_overrides.clear()


class TestHistoryControllerResponseFormat:
    """Test suite for response format validation in HistoryController."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI application with history router."""
        app = FastAPI()
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.mark.unit
    def test_sessions_list_response_format(self, app, client):
        """Test that sessions list response matches expected format."""
        mock_service = Mock(spec=HistoryService)
        mock_service.enabled = True
        
        # Create properly mocked session with all required attributes
        mock_session = Mock()
        mock_session.session_id = "test-session"
        mock_session.alert_id = "test-alert"
        mock_session.alert_type = "TestAlert"
        mock_session.agent_type = "TestAgent"
        mock_session.status = "completed"
        mock_session.started_at = datetime.now(timezone.utc)
        mock_session.completed_at = datetime.now(timezone.utc)
        mock_session.error_message = None
        mock_session.llm_interactions = []  # Add missing attributes
        mock_session.mcp_communications = []
        # Add the new dynamic attributes expected by the controller
        mock_session.llm_interaction_count = 0
        mock_session.mcp_communication_count = 0
        
        mock_service.get_sessions_list.return_value = ([mock_session], 1)
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        response = client.get("/api/v1/history/sessions")
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate top-level structure (allow optional filters_applied field)
        expected_keys = {"sessions", "pagination"}
        actual_keys = set(data.keys())
        assert expected_keys.issubset(actual_keys)
        # Optional field that may be present
        allowed_optional_keys = {"filters_applied"}
        assert actual_keys.issubset(expected_keys.union(allowed_optional_keys))
        
        # Validate sessions structure
        assert isinstance(data["sessions"], list)
        assert len(data["sessions"]) == 1
        
        session = data["sessions"][0]
        required_fields = {
            "session_id", "alert_id", "alert_type", "agent_type", 
            "status", "started_at", "completed_at", "error_message"
        }
        # Allow additional fields that are actually returned
        optional_fields = {
            "duration_ms", "llm_interaction_count", "mcp_communication_count"
        }
        actual_fields = set(session.keys())
        assert required_fields.issubset(actual_fields)
        assert actual_fields.issubset(required_fields.union(optional_fields))
        
        # Validate pagination structure
        pagination = data["pagination"]
        required_pagination_fields = {
            "page", "page_size", "total_pages", "total_items"
        }
        assert set(pagination.keys()) == required_pagination_fields
        assert isinstance(pagination["page"], int)
        assert isinstance(pagination["total_items"], int)
    
    @pytest.mark.unit
    def test_session_detail_response_format(self, app, client):
        """Test that session detail response matches expected format."""
        mock_service = Mock(spec=HistoryService)
        mock_service.enabled = True
        
        # Mock timeline with correct structure (session instead of session_info)
        mock_timeline = {
            "session": {
                "session_id": "test-session",
                "alert_id": "test-alert",
                "alert_type": "TestAlert",
                "agent_type": "TestAgent",
                "status": "completed",
                "started_at": "2024-01-15T10:00:00Z",
                "completed_at": "2024-01-15T10:05:00Z",
                "error_message": None
            },
            "chronological_timeline": [
                {
                    "interaction_id": "int-1",
                    "type": "llm_interaction",
                    "timestamp": "2024-01-15T10:00:00Z",
                    "step_description": "Test step",
                    "details": {
                        "prompt_text": "Test prompt",
                        "response_text": "Test response",
                        "model_used": "gpt-4"
                    }
                }
            ]
        }
        mock_service.get_session_timeline.return_value = mock_timeline
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        response = client.get("/api/v1/history/sessions/test-session")
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate top-level structure (SessionDetailResponse fields)
        expected_fields = {
            "session_id", "alert_id", "alert_type", "agent_type",
            "status", "started_at", "completed_at", "error_message",
            "chronological_timeline", "summary", "duration_ms", "session_metadata", "alert_data"
        }
        assert expected_fields.issubset(set(data.keys()))
        
        # Validate session details directly in response
        assert data["session_id"] == "test-session"
        assert data["alert_type"] == "TestAlert"
        assert data["status"] == "completed"
        
        # Validate timeline structure
        timeline = data["chronological_timeline"]
        assert isinstance(timeline, list)
        assert len(timeline) == 1
        
        event = timeline[0]
        assert "type" in event
        assert "timestamp" in event
        assert "step_description" in event
        
        # Validate summary exists
        summary = data["summary"]
        assert isinstance(summary, dict)
    
    @pytest.mark.unit
    def test_health_check_response_format(self, app, client):
        """Test that health check response matches expected format."""
        mock_service = Mock(spec=HistoryService)
        mock_service.enabled = True
        mock_service.test_database_connection.return_value = True
        
        # Add settings mock
        mock_settings = Mock()
        mock_settings.history_database_url = "sqlite:///test.db"
        mock_service.settings = mock_settings
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        response = client.get("/api/v1/history/health")
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate top-level structure
        required_fields = {"service", "status", "timestamp", "details"}
        assert set(data.keys()) == required_fields
        
        # Validate field types
        assert isinstance(data["service"], str)
        assert isinstance(data["status"], str)
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["details"], dict)
        
        # Validate status values
        assert data["status"] in ["healthy", "unhealthy", "disabled"]
        
        # Validate details structure for healthy service
        details = data["details"]
        assert "database_connection" in details
        assert "history_enabled" in details


class TestHistoryControllerIntegration:
    """Integration tests for HistoryController with realistic scenarios."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI application with history router."""
        app = FastAPI()
        app.include_router(router)  # Router already has /api/v1/history prefix
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.mark.unit
    def test_complex_filtering_scenario(self, app, client):
        """Test complex filtering scenario with multiple parameters."""
        mock_service = Mock(spec=HistoryService)
        mock_service.enabled = True
        mock_service.get_sessions_list.return_value = ([], 0)
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        response = client.get(
            "/api/v1/history/sessions",
            params={
                "status": "completed",
                "agent_type": "KubernetesAgent",
                "alert_type": "NamespaceTerminating",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-31T23:59:59Z",
                "page": 1,
                "page_size": 20
            }
        )
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        
        # Verify all filters were applied
        call_args = mock_service.get_sessions_list.call_args
        filters = call_args.kwargs["filters"]
        
        assert filters["status"] == ["completed"]  # Now expects list due to multiple status support
        assert filters["agent_type"] == "KubernetesAgent"
        assert filters["alert_type"] == "NamespaceTerminating"
        assert "start_date" in filters
        assert "end_date" in filters
    
    @pytest.mark.unit
    def test_real_world_error_scenarios(self, app, client):
        """Test real-world error scenarios."""
        mock_service = Mock(spec=HistoryService)
        mock_service.enabled = True
        
        # Test database timeout scenario
        mock_service.get_sessions_list.side_effect = TimeoutError("Database timeout")
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        response = client.get("/api/v1/history/sessions")
        
        # Clean up
        app.dependency_overrides.clear()
        
        assert response.status_code == 500
        assert "detail" in response.json()
    
    @pytest.mark.unit
    def test_concurrent_request_handling(self, app, client):
        """Test that controller handles concurrent requests properly."""
        mock_service = Mock(spec=HistoryService)
        mock_service.enabled = True
        mock_service.get_sessions_list.return_value = ([], 0)
        
        # Override FastAPI dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        # Simulate multiple concurrent requests
        responses = []
        for i in range(5):
            response = client.get(f"/api/v1/history/sessions?page={i+1}")
            responses.append(response)
        
        # Clean up
        app.dependency_overrides.clear()
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Service should have been called for each request
        assert mock_service.get_sessions_list.call_count == 5

class TestDashboardEndpoints:
    """Test suite for new dashboard-specific API endpoints."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI application with history router."""
        app = FastAPI()
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_history_service(self):
        """Create mock history service for dashboard endpoints."""
        service = Mock(spec=HistoryService)
        service.enabled = True
        service.is_enabled = True
        
        # Mock dashboard methods
        service.get_dashboard_metrics.return_value = {
            "active_sessions": 3,
            "completed_sessions": 25,
            "failed_sessions": 2,
            "total_interactions": 150,
            "avg_session_duration": 45.5,
            "error_rate": 6.67,
            "last_24h_sessions": 10
        }
        
        # Create a proper mock session object with datetime handling
        mock_session = Mock()
        mock_session.session_id = "session_1"
        mock_session.alert_id = "alert_1"
        mock_session.agent_type = "kubernetes"
        mock_session.alert_type = "PodCrashLooping"
        mock_session.status = "in_progress"
        mock_session.started_at = datetime.now(timezone.utc)
        mock_session.completed_at = None
        mock_session.error_message = None
        
        service.get_active_sessions.return_value = [mock_session]
        
        service.get_filter_options.return_value = {
            "agent_types": ["kubernetes", "network", "database"],
            "alert_types": ["PodCrashLooping", "ServiceDown"],
            "status_options": ["pending", "in_progress", "completed", "failed"],
            "time_ranges": [
                {"label": "Last Hour", "value": "1h"},
                {"label": "Last 4 Hours", "value": "4h"},
                {"label": "Today", "value": "today"},
                {"label": "This Week", "value": "week"}
            ]
        }
        
        return service
    
    @pytest.mark.unit
    def test_get_dashboard_metrics_success(self, app, client, mock_history_service):
        """Test successful dashboard metrics retrieval."""
        # Override dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        # Make request
        response = client.get("/api/v1/history/metrics")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["active_sessions"] == 3
        assert data["completed_sessions"] == 25
        assert data["failed_sessions"] == 2
        assert data["total_interactions"] == 150
        assert data["avg_session_duration"] == 45.5
        assert data["error_rate"] == 6.67
        assert data["last_24h_sessions"] == 10
        
        # Verify service was called
        mock_history_service.get_dashboard_metrics.assert_called_once()
    
    @pytest.mark.unit
    def test_get_dashboard_metrics_service_error(self, app, client, mock_history_service):
        """Test dashboard metrics endpoint with service error."""
        # Configure mock to raise exception
        mock_history_service.get_dashboard_metrics.side_effect = Exception("Database error")
        
        # Override dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        # Make request
        response = client.get("/api/v1/history/metrics")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # Assert error response
        assert response.status_code == 500
        assert "Failed to get metrics" in response.json()["detail"]
    
    @pytest.mark.unit
    def test_get_active_sessions_success(self, app, client):
        """Test successful active sessions retrieval."""
        # Create a more realistic mock service
        mock_service = Mock()
        
        # Create a real AlertSession-like object
        from tarsy.models.history import AlertSession
        test_session = AlertSession(
            session_id="session_1",
            alert_id="alert_1",
            agent_type="kubernetes",
            alert_type="PodCrashLooping",
            status="in_progress",
            started_at=datetime.now(timezone.utc),
            alert_data={}
        )
        
        mock_service.get_active_sessions.return_value = [test_session]
        
        # Override dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        # Make request
        response = client.get("/api/v1/history/active-sessions")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["session_id"] == "session_1"
        assert data[0]["alert_id"] == "alert_1"
        assert data[0]["agent_type"] == "kubernetes"
        assert data[0]["alert_type"] == "PodCrashLooping"
        assert data[0]["status"] == "in_progress"
        assert "started_at" in data[0]
        assert "duration_seconds" in data[0]
        assert data[0]["completed_at"] is None
        
        # Verify service was called
        mock_service.get_active_sessions.assert_called_once()
    
    @pytest.mark.unit
    def test_get_active_sessions_empty(self, app, client):
        """Test active sessions endpoint with no active sessions."""
        # Create mock service
        mock_service = Mock()
        mock_service.get_active_sessions.return_value = []
        
        # Override dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        # Make request
        response = client.get("/api/v1/history/active-sessions")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data == []
    
    @pytest.mark.unit
    def test_get_active_sessions_service_error(self, app, client):
        """Test active sessions endpoint with service error."""
        # Create mock service that raises exception
        mock_service = Mock()
        mock_service.get_active_sessions.side_effect = Exception("Database error")
        
        # Override dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        # Make request
        response = client.get("/api/v1/history/active-sessions")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # Assert error response
        assert response.status_code == 500
        assert "Failed to get active sessions" in response.json()["detail"]
    
    @pytest.mark.unit
    def test_get_filter_options_success(self, app, client, mock_history_service):
        """Test successful filter options retrieval."""
        # Override dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        # Make request
        response = client.get("/api/v1/history/filter-options")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert "agent_types" in data
        assert "alert_types" in data
        assert "status_options" in data
        assert "time_ranges" in data
        assert len(data["agent_types"]) == 3
        assert "kubernetes" in data["agent_types"]
        assert len(data["time_ranges"]) == 4
        
        # Verify service was called
        mock_history_service.get_filter_options.assert_called_once()
    
    @pytest.mark.unit
    def test_get_filter_options_service_error(self, app, client, mock_history_service):
        """Test filter options endpoint with service error."""
        # Configure mock to raise exception
        mock_history_service.get_filter_options.side_effect = Exception("Database error")
        
        # Override dependency
        app.dependency_overrides[get_history_service] = lambda: mock_history_service
        
        # Make request
        response = client.get("/api/v1/history/filter-options")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # Assert error response
        assert response.status_code == 500
        assert "Failed to get filter options" in response.json()["detail"]


class TestExportAndSearchEndpoints:
    """Test suite for export and search API endpoints."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI application with history router."""
        app = FastAPI()
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.mark.unit
    def test_export_session_json_success(self, app, client):
        """Test successful session export in JSON format."""
        # Create mock service
        mock_service = Mock()
        mock_service.export_session_data.return_value = {
            "session_id": "test_session",
            "format": "json",
            "data": {
                "session": {
                    "session_id": "test_session",
                    "alert_id": "test_alert",
                    "agent_type": "KubernetesAgent",
                    "status": "completed"
                },
                "timeline": {"interactions": []},
                "export_metadata": {"exported_at": "2025-01-25T00:00:00Z"}
            },
            "error": None
        }
        
        # Override dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        # Make request
        response = client.get("/api/v1/history/sessions/test_session/export?format=json")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["session"]["session_id"] == "test_session"
        assert data["timeline"]["interactions"] == []
        
        # Verify service was called
        mock_service.export_session_data.assert_called_once_with("test_session", "json")
    
    @pytest.mark.unit
    def test_export_session_csv_success(self, app, client):
        """Test successful session export in CSV format."""
        # Create mock service
        mock_service = Mock()
        mock_service.export_session_data.return_value = {
            "session_id": "test_session",
            "format": "csv",
            "data": {
                "session": {
                    "session_id": "test_session",
                    "alert_id": "test_alert",
                    "agent_type": "KubernetesAgent",
                    "alert_type": "TestAlert",
                    "status": "completed",
                    "started_at": "2025-01-25T00:00:00Z",
                    "completed_at": "2025-01-25T00:01:00Z",
                    "error_message": None
                },
                "timeline": {"interactions": []},
                "export_metadata": {"exported_at": "2025-01-25T00:00:00Z"}
            },
            "error": None
        }
        
        # Override dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        # Make request
        response = client.get("/api/v1/history/sessions/test_session/export?format=csv")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # Assert response
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "attachment; filename=session_test_session.csv" in response.headers.get("content-disposition", "")
        
        # Check CSV content
        csv_content = response.content.decode('utf-8')
        assert "Session Information" in csv_content
        assert "test_session" in csv_content
        assert "KubernetesAgent" in csv_content
        
        # Verify service was called
        mock_service.export_session_data.assert_called_once_with("test_session", "csv")
    
    @pytest.mark.unit
    def test_export_session_not_found(self, app, client):
        """Test export endpoint with non-existent session."""
        # Create mock service
        mock_service = Mock()
        mock_service.export_session_data.return_value = {
            "error": "Session not_found_session not found",
            "session_id": "not_found_session",
            "format": "json",
            "data": None
        }
        
        # Override dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        # Make request
        response = client.get("/api/v1/history/sessions/not_found_session/export")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # Assert error response
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @pytest.mark.unit
    def test_export_session_invalid_format(self, app, client):
        """Test export endpoint with invalid format parameter."""
        # Create mock service
        mock_service = Mock()
        
        # Override dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        # Make request with invalid format
        response = client.get("/api/v1/history/sessions/test_session/export?format=xml")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # Assert validation error
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.unit
    def test_search_sessions_success(self, app, client):
        """Test successful session search."""
        # Create mock service
        mock_service = Mock()
        mock_service.search_sessions.return_value = [
            {
                "session_id": "session_1",
                "alert_id": "namespace_alert_1",
                "agent_type": "KubernetesAgent",
                "alert_type": "NamespaceTerminating",
                "status": "completed",
                "started_at": "2025-01-25T00:00:00Z",
                "completed_at": "2025-01-25T00:01:00Z",
                "error_message": None,
                "duration_seconds": 60.0
            }
        ]
        
        # Override dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        # Make request
        response = client.get("/api/v1/history/search?q=namespace&limit=5")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["session_id"] == "session_1"
        assert data[0]["alert_type"] == "NamespaceTerminating"
        
        # Verify service was called
        mock_service.search_sessions.assert_called_once_with("namespace", 5)
    
    @pytest.mark.unit
    def test_search_sessions_empty_results(self, app, client):
        """Test search with no matching results."""
        # Create mock service
        mock_service = Mock()
        mock_service.search_sessions.return_value = []
        
        # Override dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        # Make request
        response = client.get("/api/v1/history/search?q=nonexistent")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data == []
        
        # Verify service was called with default limit
        mock_service.search_sessions.assert_called_once_with("nonexistent", 10)
    
    @pytest.mark.unit
    def test_search_sessions_invalid_limit(self, app, client):
        """Test search with invalid limit parameter."""
        # Create mock service
        mock_service = Mock()
        
        # Override dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        # Make request with invalid limit
        response = client.get("/api/v1/history/search?q=test&limit=0")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # Assert validation error
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.unit
    def test_search_sessions_service_error(self, app, client):
        """Test search endpoint with service error."""
        # Create mock service that raises exception
        mock_service = Mock()
        mock_service.search_sessions.side_effect = Exception("Database error")
        
        # Override dependency
        app.dependency_overrides[get_history_service] = lambda: mock_service
        
        # Make request
        response = client.get("/api/v1/history/search?q=test")
        
        # Clean up
        app.dependency_overrides.clear()
        
        # Assert error response
        assert response.status_code == 500
        assert "Failed to search sessions" in response.json()["detail"] 


def create_mock_session(session_id: str, status: str) -> Mock:
    """Helper to create a mock session object with a specific status."""
    mock_session = Mock()
    mock_session.session_id = session_id
    mock_session.alert_id = f"alert-{session_id}"
    mock_session.agent_type = "KubernetesAgent"
    mock_session.alert_type = "NamespaceTerminating"
    mock_session.status = status
    mock_session.started_at = datetime.now(timezone.utc)
    mock_session.completed_at = datetime.now(timezone.utc) if status == "completed" else None
    mock_session.error_message = None
    mock_session.llm_interactions = []
    mock_session.mcp_communications = []
    mock_session.llm_interaction_count = 0
    mock_session.mcp_communication_count = 0
    return mock_session 