import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.coordinator import (
    _get_mcp_url,
    _get_hf_token,
    _create_model,
    run_deep_research,
    COORDINATOR_MODEL_ID,
    SUBAGENT_MODEL_ID,
)
from src.task_splitter import Subtask


class TestGetMcpUrl:
    """Tests for the _get_mcp_url helper function."""

    def test_returns_correct_url(self, monkeypatch):
        """Test that the correct MCP URL is constructed."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-api-key")
        result = _get_mcp_url()
        assert result == "https://mcp.firecrawl.dev/test-api-key/v2/mcp"

    def test_missing_api_key_raises_error(self, monkeypatch):
        """Test that missing FIRECRAWL_API_KEY raises KeyError."""
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        with pytest.raises(KeyError, match="FIRECRAWL_API_KEY environment variable is not set"):
            _get_mcp_url()


class TestGetHfToken:
    """Tests for the _get_hf_token helper function."""

    def test_returns_token(self, monkeypatch):
        """Test that the HF token is returned correctly."""
        monkeypatch.setenv("HF_TOKEN", "test-hf-token")
        result = _get_hf_token()
        assert result == "test-hf-token"

    def test_missing_token_raises_error(self, monkeypatch):
        """Test that missing HF_TOKEN raises KeyError."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(KeyError, match="HF_TOKEN environment variable is not set"):
            _get_hf_token()


class TestCreateModel:
    """Tests for the _create_model helper function."""

    @patch("src.coordinator.InferenceClientModel")
    def test_creates_model_with_correct_params(self, mock_model_class):
        """Test that the model is created with correct parameters."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        result = _create_model("test-model-id", "test-api-key")

        mock_model_class.assert_called_once_with(
            model_id="test-model-id",
            api_key="test-api-key",
        )
        assert result == mock_model


class TestRunDeepResearch:
    """Tests for the run_deep_research function."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Set up required environment variables."""
        monkeypatch.setenv("HF_TOKEN", "test-hf-token")
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-firecrawl-key")

    @pytest.fixture
    def mock_subtasks(self):
        """Create mock subtasks for testing."""
        return [
            Subtask(
                id="history",
                title="Historical Background",
                description="Research the historical context.",
            ),
            Subtask(
                id="current",
                title="Current State",
                description="Analyze the current situation.",
            ),
        ]

    @pytest.fixture
    def mock_research_plan(self):
        """Create a mock research plan."""
        return "1. Research history\n2. Analyze current state\n3. Summarize findings"

    @patch("src.coordinator.MCPClient")
    @patch("src.coordinator._create_model")
    @patch("src.coordinator.split_into_subtasks")
    @patch("src.coordinator.generate_research_plan")
    def test_run_deep_research_success(
        self,
        mock_generate_plan,
        mock_split_subtasks,
        mock_create_model,
        mock_mcp_client,
        mock_env,
        mock_subtasks,
        mock_research_plan,
    ):
        """Test successful deep research execution."""
        # Arrange
        mock_generate_plan.return_value = mock_research_plan
        mock_split_subtasks.return_value = mock_subtasks

        mock_coordinator_model = Mock()
        mock_subagent_model = Mock()
        mock_create_model.side_effect = [mock_coordinator_model, mock_subagent_model]

        mock_mcp_tools = Mock()
        mock_mcp_client.return_value.__enter__ = Mock(return_value=mock_mcp_tools)
        mock_mcp_client.return_value.__exit__ = Mock(return_value=False)

        # Mock the coordinator agent's run method
        with patch("src.coordinator.ToolCallingAgent") as mock_agent_class:
            mock_coordinator = Mock()
            mock_coordinator.run.return_value = "# Final Research Report\n\nFindings here."
            mock_agent_class.return_value = mock_coordinator

            # Act
            result = run_deep_research("What is quantum computing?")

            # Assert
            assert result == "# Final Research Report\n\nFindings here."
            mock_generate_plan.assert_called_once_with("What is quantum computing?")
            mock_split_subtasks.assert_called_once_with(mock_research_plan)

    @patch("src.coordinator.MCPClient")
    @patch("src.coordinator._create_model")
    @patch("src.coordinator.split_into_subtasks")
    @patch("src.coordinator.generate_research_plan")
    def test_run_deep_research_creates_correct_models(
        self,
        mock_generate_plan,
        mock_split_subtasks,
        mock_create_model,
        mock_mcp_client,
        mock_env,
        mock_subtasks,
        mock_research_plan,
    ):
        """Test that coordinator and subagent models are created correctly."""
        # Arrange
        mock_generate_plan.return_value = mock_research_plan
        mock_split_subtasks.return_value = mock_subtasks
        mock_create_model.return_value = Mock()

        mock_mcp_client.return_value.__enter__ = Mock(return_value=Mock())
        mock_mcp_client.return_value.__exit__ = Mock(return_value=False)

        with patch("src.coordinator.ToolCallingAgent") as mock_agent_class:
            mock_agent_class.return_value.run.return_value = "Report"

            # Act
            run_deep_research("Test query")

            # Assert
            assert mock_create_model.call_count == 2
            calls = mock_create_model.call_args_list
            assert calls[0][0] == (COORDINATOR_MODEL_ID, "test-hf-token")
            assert calls[1][0] == (SUBAGENT_MODEL_ID, "test-hf-token")

    @patch("src.coordinator.MCPClient")
    @patch("src.coordinator._create_model")
    @patch("src.coordinator.split_into_subtasks")
    @patch("src.coordinator.generate_research_plan")
    def test_run_deep_research_uses_correct_mcp_url(
        self,
        mock_generate_plan,
        mock_split_subtasks,
        mock_create_model,
        mock_mcp_client,
        mock_env,
        mock_subtasks,
        mock_research_plan,
    ):
        """Test that the correct MCP URL is used."""
        # Arrange
        mock_generate_plan.return_value = mock_research_plan
        mock_split_subtasks.return_value = mock_subtasks
        mock_create_model.return_value = Mock()

        mock_mcp_client.return_value.__enter__ = Mock(return_value=Mock())
        mock_mcp_client.return_value.__exit__ = Mock(return_value=False)

        with patch("src.coordinator.ToolCallingAgent") as mock_agent_class:
            mock_agent_class.return_value.run.return_value = "Report"

            # Act
            run_deep_research("Test query")

            # Assert
            mock_mcp_client.assert_called_once_with({
                "url": "https://mcp.firecrawl.dev/test-firecrawl-key/v2/mcp",
                "transport": "streamable-http",
            })

    @patch("src.coordinator.MCPClient")
    @patch("src.coordinator._create_model")
    @patch("src.coordinator.split_into_subtasks")
    @patch("src.coordinator.generate_research_plan")
    def test_run_deep_research_coordinator_prompt_format(
        self,
        mock_generate_plan,
        mock_split_subtasks,
        mock_create_model,
        mock_mcp_client,
        mock_env,
        mock_subtasks,
        mock_research_plan,
    ):
        """Test that the coordinator prompt is formatted correctly."""
        # Arrange
        mock_generate_plan.return_value = mock_research_plan
        mock_split_subtasks.return_value = mock_subtasks
        mock_create_model.return_value = Mock()

        mock_mcp_client.return_value.__enter__ = Mock(return_value=Mock())
        mock_mcp_client.return_value.__exit__ = Mock(return_value=False)

        with patch("src.coordinator.ToolCallingAgent") as mock_agent_class:
            mock_coordinator = Mock()
            mock_agent_class.return_value = mock_coordinator
            mock_coordinator.run.return_value = "Report"

            # Act
            run_deep_research("What is AI?")

            # Assert
            prompt_arg = mock_coordinator.run.call_args[0][0]
            assert "What is AI?" in prompt_arg
            assert mock_research_plan in prompt_arg
            # Check subtasks are JSON serialized in the prompt
            assert "history" in prompt_arg
            assert "Historical Background" in prompt_arg

    @patch("src.coordinator.MCPClient")
    @patch("src.coordinator._create_model")
    @patch("src.coordinator.split_into_subtasks")
    @patch("src.coordinator.generate_research_plan")
    def test_run_deep_research_prints_status(
        self,
        mock_generate_plan,
        mock_split_subtasks,
        mock_create_model,
        mock_mcp_client,
        mock_env,
        mock_subtasks,
        mock_research_plan,
        capsys,
    ):
        """Test that status messages are printed during execution."""
        # Arrange
        mock_generate_plan.return_value = mock_research_plan
        mock_split_subtasks.return_value = mock_subtasks
        mock_create_model.return_value = Mock()

        mock_mcp_client.return_value.__enter__ = Mock(return_value=Mock())
        mock_mcp_client.return_value.__exit__ = Mock(return_value=False)

        with patch("src.coordinator.ToolCallingAgent") as mock_agent_class:
            mock_agent_class.return_value.run.return_value = "Report"

            # Act
            run_deep_research("Test query")

            # Assert
            captured = capsys.readouterr()
            assert "*** Running the deep research ***" in captured.out
            assert "*** Initializing Coordinator ***" in captured.out

    def test_missing_hf_token_raises_error(self, monkeypatch):
        """Test that missing HF_TOKEN raises KeyError."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")

        with patch("src.coordinator.generate_research_plan") as mock_plan:
            with patch("src.coordinator.split_into_subtasks") as mock_split:
                mock_plan.return_value = "plan"
                mock_split.return_value = []

                with pytest.raises(KeyError, match="HF_TOKEN"):
                    run_deep_research("Test")

    def test_missing_firecrawl_key_raises_error(self, monkeypatch):
        """Test that missing FIRECRAWL_API_KEY raises KeyError."""
        monkeypatch.setenv("HF_TOKEN", "test-token")
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)

        with patch("src.coordinator.generate_research_plan") as mock_plan:
            with patch("src.coordinator.split_into_subtasks") as mock_split:
                with patch("src.coordinator._create_model"):
                    mock_plan.return_value = "plan"
                    mock_split.return_value = []

                    with pytest.raises(KeyError, match="FIRECRAWL_API_KEY"):
                        run_deep_research("Test")

    @patch("src.coordinator.MCPClient")
    @patch("src.coordinator._create_model")
    @patch("src.coordinator.split_into_subtasks")
    @patch("src.coordinator.generate_research_plan")
    def test_run_deep_research_coordinator_agent_configuration(
        self,
        mock_generate_plan,
        mock_split_subtasks,
        mock_create_model,
        mock_mcp_client,
        mock_env,
        mock_subtasks,
        mock_research_plan,
    ):
        """Test that the coordinator agent is configured correctly."""
        # Arrange
        mock_generate_plan.return_value = mock_research_plan
        mock_split_subtasks.return_value = mock_subtasks

        mock_coordinator_model = Mock()
        mock_subagent_model = Mock()
        mock_create_model.side_effect = [mock_coordinator_model, mock_subagent_model]

        mock_mcp_client.return_value.__enter__ = Mock(return_value=Mock())
        mock_mcp_client.return_value.__exit__ = Mock(return_value=False)

        with patch("src.coordinator.ToolCallingAgent") as mock_agent_class:
            mock_agent_class.return_value.run.return_value = "Report"

            # Act
            run_deep_research("Test query")

            # Assert
            # The coordinator agent should be created with the initialize_subagent tool
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs["model"] == mock_coordinator_model
            assert call_kwargs["add_base_tools"] is False
            assert call_kwargs["name"] == "coordinator_agent"
            assert len(call_kwargs["tools"]) == 1  # initialize_subagent tool

    @patch("src.coordinator.MCPClient")
    @patch("src.coordinator._create_model")
    @patch("src.coordinator.split_into_subtasks")
    @patch("src.coordinator.generate_research_plan")
    def test_run_deep_research_subtasks_json_serialization(
        self,
        mock_generate_plan,
        mock_split_subtasks,
        mock_create_model,
        mock_mcp_client,
        mock_env,
        mock_subtasks,
        mock_research_plan,
    ):
        """Test that subtasks are correctly serialized to JSON."""
        # Arrange
        mock_generate_plan.return_value = mock_research_plan
        mock_split_subtasks.return_value = mock_subtasks
        mock_create_model.return_value = Mock()

        mock_mcp_client.return_value.__enter__ = Mock(return_value=Mock())
        mock_mcp_client.return_value.__exit__ = Mock(return_value=False)

        with patch("src.coordinator.ToolCallingAgent") as mock_agent_class:
            mock_coordinator = Mock()
            mock_agent_class.return_value = mock_coordinator
            mock_coordinator.run.return_value = "Report"

            # Act
            run_deep_research("Test query")

            # Assert
            prompt_arg = mock_coordinator.run.call_args[0][0]
            # The prompt should contain valid JSON for subtasks
            expected_subtasks = [
                {"id": "history", "title": "Historical Background", "description": "Research the historical context."},
                {"id": "current", "title": "Current State", "description": "Analyze the current situation."},
            ]
            for subtask in expected_subtasks:
                assert subtask["id"] in prompt_arg
                assert subtask["title"] in prompt_arg


class TestInitializeSubagentTool:
    """Tests for the initialize_subagent tool defined inside run_deep_research."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Set up required environment variables."""
        monkeypatch.setenv("HF_TOKEN", "test-hf-token")
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-firecrawl-key")

    @patch("src.coordinator.MCPClient")
    @patch("src.coordinator._create_model")
    @patch("src.coordinator.split_into_subtasks")
    @patch("src.coordinator.generate_research_plan")
    def test_initialize_subagent_tool_is_created(
        self,
        mock_generate_plan,
        mock_split_subtasks,
        mock_create_model,
        mock_mcp_client,
        mock_env,
    ):
        """Test that the initialize_subagent tool is created and passed to coordinator."""
        # Arrange
        mock_generate_plan.return_value = "research plan"
        mock_split_subtasks.return_value = []
        mock_create_model.return_value = Mock()

        mock_mcp_client.return_value.__enter__ = Mock(return_value=Mock())
        mock_mcp_client.return_value.__exit__ = Mock(return_value=False)

        with patch("src.coordinator.ToolCallingAgent") as mock_agent_class:
            mock_agent_class.return_value.run.return_value = "Report"

            # Act
            run_deep_research("Test query")

            # Assert
            call_kwargs = mock_agent_class.call_args[1]
            tools = call_kwargs["tools"]
            assert len(tools) == 1
            # The tool should be a callable
            assert callable(tools[0])


class TestModelConstants:
    """Tests for module-level constants."""

    def test_coordinator_model_id_is_defined(self):
        """Test that COORDINATOR_MODEL_ID is defined."""
        assert COORDINATOR_MODEL_ID == "MiniMaxAI/MiniMax-M1-80k"

    def test_subagent_model_id_is_defined(self):
        """Test that SUBAGENT_MODEL_ID is defined."""
        assert SUBAGENT_MODEL_ID == "MiniMaxAI/MiniMax-M1-80k"

    def test_model_ids_are_same(self):
        """Test that both models use the same ID (current implementation)."""
        assert COORDINATOR_MODEL_ID == SUBAGENT_MODEL_ID
