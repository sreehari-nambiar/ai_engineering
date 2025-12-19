import pytest
import json
from unittest.mock import Mock, patch

from src.task_splitter import Subtask, SubtaskList, split_into_subtasks, MODEL_ID


class TestSubtaskModel:
    """Tests for the Subtask Pydantic model."""

    def test_subtask_valid_creation(self):
        """Test creating a valid Subtask instance."""
        subtask = Subtask(
            id="A",
            title="Research History",
            description="Research the historical background of the topic.",
        )
        assert subtask.id == "A"
        assert subtask.title == "Research History"
        assert subtask.description == "Research the historical background of the topic."

    def test_subtask_missing_required_field(self):
        """Test that missing required fields raise validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Subtask(id="A", title="Test")  # missing description

    def test_subtask_to_dict(self):
        """Test converting Subtask to dictionary."""
        subtask = Subtask(id="B", title="Analysis", description="Analyze the data.")
        data = subtask.model_dump()
        assert data == {
            "id": "B",
            "title": "Analysis",
            "description": "Analyze the data.",
        }


class TestSubtaskListModel:
    """Tests for the SubtaskList Pydantic model."""

    def test_subtask_list_valid_creation(self):
        """Test creating a valid SubtaskList instance."""
        subtask_list = SubtaskList(
            subtasks=[
                Subtask(id="A", title="Task A", description="Description A"),
                Subtask(id="B", title="Task B", description="Description B"),
            ]
        )
        assert len(subtask_list.subtasks) == 2
        assert subtask_list.subtasks[0].id == "A"
        assert subtask_list.subtasks[1].id == "B"

    def test_subtask_list_empty(self):
        """Test creating an empty SubtaskList."""
        subtask_list = SubtaskList(subtasks=[])
        assert len(subtask_list.subtasks) == 0

    def test_subtask_list_json_schema(self):
        """Test that JSON schema is generated correctly."""
        schema = SubtaskList.model_json_schema()
        assert "properties" in schema
        assert "subtasks" in schema["properties"]


class TestSplitIntoSubtasks:
    """Tests for the split_into_subtasks function."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Set up HF_TOKEN environment variable."""
        monkeypatch.setenv("HF_TOKEN", "test-token")

    @pytest.fixture
    def mock_api_response(self):
        """Create a mock API response with subtasks."""
        return {
            "subtasks": [
                {
                    "id": "history",
                    "title": "Historical Background",
                    "description": "Research the historical context and origins.",
                },
                {
                    "id": "current",
                    "title": "Current State",
                    "description": "Analyze the current state of affairs.",
                },
                {
                    "id": "future",
                    "title": "Future Outlook",
                    "description": "Predict future trends and developments.",
                },
            ]
        }

    @patch("src.task_splitter.InferenceClient")
    def test_split_into_subtasks_success(
        self, mock_client_class, mock_env, mock_api_response
    ):
        """Test successful splitting of research plan into subtasks."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = json.dumps(mock_api_response)

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = mock_message

        mock_client.chat.completions.create.return_value = mock_completion

        # Act
        result = split_into_subtasks("Research quantum computing applications")

        # Assert
        assert len(result) == 3
        assert all(isinstance(subtask, Subtask) for subtask in result)
        assert result[0].id == "history"
        assert result[0].title == "Historical Background"
        assert result[1].id == "current"
        assert result[2].id == "future"

    @patch("src.task_splitter.InferenceClient")
    def test_split_into_subtasks_uses_correct_model(
        self, mock_client_class, mock_env, mock_api_response
    ):
        """Test that the correct model ID is used."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = json.dumps(mock_api_response)

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = mock_message

        mock_client.chat.completions.create.return_value = mock_completion

        # Act
        split_into_subtasks("Test research plan")

        # Assert
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == MODEL_ID

    @patch("src.task_splitter.InferenceClient")
    def test_split_into_subtasks_custom_model(
        self, mock_client_class, mock_env, mock_api_response
    ):
        """Test using a custom model ID."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = json.dumps(mock_api_response)

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = mock_message

        mock_client.chat.completions.create.return_value = mock_completion

        custom_model = "custom/model-id"

        # Act
        split_into_subtasks("Test research plan", model_id=custom_model)

        # Assert
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == custom_model

    @patch("src.task_splitter.InferenceClient")
    def test_split_into_subtasks_uses_hf_token(
        self, mock_client_class, mock_env, mock_api_response
    ):
        """Test that HF_TOKEN is passed to the client."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = json.dumps(mock_api_response)

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = mock_message

        mock_client.chat.completions.create.return_value = mock_completion

        # Act
        split_into_subtasks("Test")

        # Assert
        mock_client_class.assert_called_once_with(api_key="test-token")

    @patch("src.task_splitter.InferenceClient")
    def test_split_into_subtasks_uses_json_schema(
        self, mock_client_class, mock_env, mock_api_response
    ):
        """Test that JSON schema response format is used."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = json.dumps(mock_api_response)

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = mock_message

        mock_client.chat.completions.create.return_value = mock_completion

        # Act
        split_into_subtasks("Test")

        # Assert
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert "json_schema" in call_kwargs["response_format"]

    @patch("src.task_splitter.InferenceClient")
    def test_split_into_subtasks_prints_output(
        self, mock_client_class, mock_env, mock_api_response, capsys
    ):
        """Test that the function prints subtask information."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = json.dumps(mock_api_response)

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = mock_message

        mock_client.chat.completions.create.return_value = mock_completion

        # Act
        split_into_subtasks("Test research plan")

        # Assert
        captured = capsys.readouterr()
        assert "Splitting the research plan into subtasks..." in captured.out
        assert "Generated The Following Subtasks" in captured.out
        assert "Historical Background" in captured.out
        assert "Current State" in captured.out
        assert "Future Outlook" in captured.out

    def test_missing_hf_token_raises_error(self, monkeypatch):
        """Test that missing HF_TOKEN raises KeyError."""
        # Arrange
        monkeypatch.delenv("HF_TOKEN", raising=False)

        # Act & Assert
        with pytest.raises(KeyError, match="HF_TOKEN environment variable is not set"):
            split_into_subtasks("Test")

    @patch("src.task_splitter.InferenceClient")
    def test_invalid_json_response_raises_error(self, mock_client_class, mock_env):
        """Test that invalid JSON response raises ValueError."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = "not valid json"

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = mock_message

        mock_client.chat.completions.create.return_value = mock_completion

        # Act & Assert
        with pytest.raises(ValueError, match="Failed to parse subtasks from response"):
            split_into_subtasks("Test")

    @patch("src.task_splitter.InferenceClient")
    def test_missing_subtasks_key_raises_error(self, mock_client_class, mock_env):
        """Test that missing 'subtasks' key raises ValueError."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = json.dumps({"wrong_key": []})

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = mock_message

        mock_client.chat.completions.create.return_value = mock_completion

        # Act & Assert
        with pytest.raises(ValueError, match="Failed to parse subtasks from response"):
            split_into_subtasks("Test")

    @patch("src.task_splitter.InferenceClient")
    def test_empty_subtasks_list(self, mock_client_class, mock_env):
        """Test handling of empty subtasks list."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = json.dumps({"subtasks": []})

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = mock_message

        mock_client.chat.completions.create.return_value = mock_completion

        # Act
        result = split_into_subtasks("Test")

        # Assert
        assert result == []

    @patch("src.task_splitter.InferenceClient")
    def test_system_prompt_is_included(self, mock_client_class, mock_env, mock_api_response):
        """Test that the system prompt is included in messages."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = json.dumps(mock_api_response)

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = mock_message

        mock_client.chat.completions.create.return_value = mock_completion

        from src.prompt import TASK_SPLITTER_SYSTEM_INSTRUCTIONS

        # Act
        split_into_subtasks("Test query")

        # Assert
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == TASK_SPLITTER_SYSTEM_INSTRUCTIONS
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test query"

    @patch("src.task_splitter.InferenceClient")
    def test_subtask_validation_on_response(self, mock_client_class, mock_env):
        """Test that invalid subtask data raises validation error."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Response with missing required fields
        mock_message = Mock()
        mock_message.content = json.dumps({
            "subtasks": [
                {"id": "A"}  # missing title and description
            ]
        })

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = mock_message

        mock_client.chat.completions.create.return_value = mock_completion

        from pydantic import ValidationError

        # Act & Assert
        with pytest.raises(ValidationError):
            split_into_subtasks("Test")
