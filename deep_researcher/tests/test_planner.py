import pytest
from unittest.mock import Mock, patch


class TestGenerateResearchPlan:
    """Tests for the generate_research_plan function."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Set up HF_TOKEN environment variable."""
        monkeypatch.setenv("HF_TOKEN", "test-token")

    @pytest.fixture
    def mock_streaming_response(self):
        """Create a mock streaming response."""
        chunks = []
        for text in ["Step 1: ", "Research the topic. ", "Step 2: ", "Compile findings."]:
            chunk = Mock()
            chunk.choices = [Mock()]
            chunk.choices[0].delta = Mock()
            chunk.choices[0].delta.content = text
            chunks.append(chunk)
        return chunks

    @patch("src.planner.InferenceClient")
    def test_generate_research_plan_streaming(
        self, mock_client_class, mock_env, mock_streaming_response
    ):
        """Test that streaming responses are handled correctly."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter(mock_streaming_response)

        from src.planner import generate_research_plan

        # Act
        result = generate_research_plan("What is quantum computing?")

        # Assert
        assert result == "Step 1: Research the topic. Step 2: Compile findings."
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True
        assert call_kwargs["model"] == "moonshotai/Kimi-K2-Thinking"
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][1]["content"] == "What is quantum computing?"

    @patch("src.planner.InferenceClient")
    def test_generate_research_plan_non_streaming(self, mock_client_class, mock_env):
        """Test fallback to non-streaming response handling."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create a class that raises TypeError when iterated (like a real non-streaming response)
        class NonIterableResponse:
            def __init__(self):
                # Use spec to prevent auto-creation of delta attribute
                choice = Mock(spec=["message"])
                choice.message = Mock(spec=["content"])
                choice.message.content = "Complete research plan here."
                self.choices = [choice]

            def __iter__(self):
                raise TypeError("not iterable")

        mock_client.chat.completions.create.return_value = NonIterableResponse()

        from src.planner import generate_research_plan

        # Act
        result = generate_research_plan("Explain machine learning")

        # Assert
        assert result == "Complete research plan here."

    @patch("src.planner.InferenceClient")
    def test_generate_research_plan_prints_output(
        self, mock_client_class, mock_env, mock_streaming_response, capsys
    ):
        """Test that the function prints the research plan."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter(mock_streaming_response)

        from src.planner import generate_research_plan

        # Act
        generate_research_plan("Test query")

        # Assert
        captured = capsys.readouterr()
        assert "User request: Test query" in captured.out
        assert "Generated Research Plan:" in captured.out
        assert "Step 1:" in captured.out

    @patch("src.planner.InferenceClient")
    def test_generate_research_plan_uses_hf_token(self, mock_client_class, mock_env):
        """Test that the HF_TOKEN is passed to the client."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter([])

        from src.planner import generate_research_plan

        # Act
        generate_research_plan("Test")

        # Assert
        mock_client_class.assert_called_once_with(api_key="test-token")

    @patch("src.planner.InferenceClient")
    def test_generate_research_plan_handles_empty_content(
        self, mock_client_class, mock_env
    ):
        """Test handling of chunks with None content."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create chunks with some None content
        chunks = []
        for text in ["Hello", None, "World"]:
            chunk = Mock()
            chunk.choices = [Mock()]
            chunk.choices[0].delta = Mock()
            chunk.choices[0].delta.content = text
            chunks.append(chunk)

        mock_client.chat.completions.create.return_value = iter(chunks)

        from src.planner import generate_research_plan

        # Act
        result = generate_research_plan("Test")

        # Assert
        assert result == "HelloWorld"

    def test_missing_hf_token_raises_error(self, monkeypatch):
        """Test that missing HF_TOKEN raises KeyError."""
        # Arrange
        monkeypatch.delenv("HF_TOKEN", raising=False)

        # Need to reimport to avoid cached token
        import importlib
        import src.planner

        importlib.reload(src.planner)

        # Act & Assert
        with pytest.raises(KeyError):
            src.planner.generate_research_plan("Test")

    @patch("src.planner.InferenceClient")
    def test_system_prompt_is_included(self, mock_client_class, mock_env):
        """Test that the system prompt is included in messages."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter([])

        from src.planner import generate_research_plan
        from src.prompt import PLANNER_SYSTEM_INSTRUCTIONS

        # Act
        generate_research_plan("Test query")

        # Assert
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == PLANNER_SYSTEM_INSTRUCTIONS
        assert messages[1]["role"] == "user"
