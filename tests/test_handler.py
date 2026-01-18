"""Tests for the event handler."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from wyoming.asr import Transcribe
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info

from wyoming_mlx_whisper.handler import WhisperEventHandler, _pcm_to_float


class TestPcmToFloat:
    """Tests for _pcm_to_float function."""

    def test_silence(self) -> None:
        """Test conversion of silence (zeros)."""
        audio_bytes = bytes(100)  # 50 samples of silence (16-bit = 2 bytes)
        result = _pcm_to_float(audio_bytes)
        assert result.dtype == np.float32
        assert len(result) == 50
        assert np.allclose(result, 0.0)

    def test_max_positive(self) -> None:
        """Test conversion of maximum positive value."""
        # int16 max is 32767
        audio_bytes = np.array([32767], dtype=np.int16).tobytes()
        result = _pcm_to_float(audio_bytes)
        assert result.dtype == np.float32
        assert len(result) == 1
        assert np.isclose(result[0], 32767 / 32768.0)

    def test_max_negative(self) -> None:
        """Test conversion of maximum negative value."""
        # int16 min is -32768
        audio_bytes = np.array([-32768], dtype=np.int16).tobytes()
        result = _pcm_to_float(audio_bytes)
        assert result.dtype == np.float32
        assert len(result) == 1
        assert np.isclose(result[0], -1.0)

    def test_normalization_range(self) -> None:
        """Test that output is normalized to [-1, 1]."""
        # Create random int16 samples
        rng = np.random.default_rng(42)
        samples = rng.integers(-32768, 32767, size=1000, dtype=np.int16)
        audio_bytes = samples.tobytes()
        result = _pcm_to_float(audio_bytes)
        assert result.min() >= -1.0
        assert result.max() <= 1.0


class TestWhisperEventHandler:
    """Tests for WhisperEventHandler class."""

    @pytest.fixture
    def mock_wyoming_info(self) -> Info:
        """Create a mock Wyoming info object."""
        return MagicMock(spec=Info)

    @pytest.fixture
    def mock_model(self) -> str:
        """Create mock model name."""
        return "mlx-community/whisper-large-v3-turbo"

    @pytest.fixture
    def handler(
        self,
        mock_wyoming_info: Info,
        mock_model: str,
    ) -> WhisperEventHandler:
        """Create a handler instance for testing."""
        handler = WhisperEventHandler(
            mock_wyoming_info,
            mock_model,
            language=None,
            reader=MagicMock(),
            writer=MagicMock(),
        )
        handler.write_event = AsyncMock()
        return handler

    def test_init(
        self,
        mock_wyoming_info: Info,
        mock_model: str,
    ) -> None:
        """Test handler initialization."""
        handler = WhisperEventHandler(
            mock_wyoming_info,
            mock_model,
            language="en",
            reader=MagicMock(),
            writer=MagicMock(),
        )
        assert handler._model == "mlx-community/whisper-large-v3-turbo"
        assert handler._language == "en"
        assert handler._audio == b""

    def test_reset(self, handler: WhisperEventHandler) -> None:
        """Test audio buffer and context reset."""
        handler._audio = b"some audio data"
        handler._initial_prompt = "test prompt"
        handler._reset()
        assert handler._audio == b""
        assert handler._initial_prompt is None

    @pytest.mark.asyncio
    async def test_handle_audio_chunk(self, handler: WhisperEventHandler) -> None:
        """Test handling of audio chunks."""
        # Create a mock audio chunk event
        chunk = AudioChunk(
            rate=16000,
            width=2,
            channels=1,
            audio=b"\x00\x00" * 100,
        )
        event = chunk.event()

        result = await handler.handle_event(event)

        assert result is True
        assert len(handler._audio) > 0

    @pytest.mark.asyncio
    async def test_handle_audio_stop(self, handler: WhisperEventHandler) -> None:
        """Test handling of audio stop event."""
        # Pre-fill some audio data
        handler._audio = np.zeros(16000, dtype=np.int16).tobytes()  # 1 second

        # Mock the transcription
        with patch.object(
            handler,
            "_transcribe",
            return_value="Hello world",
        ):
            event = AudioStop().event()
            result = await handler.handle_event(event)

        assert result is False
        assert handler._audio == b""  # Should be reset
        handler.write_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_describe(self, handler: WhisperEventHandler) -> None:
        """Test handling of describe event."""
        event = Describe().event()
        result = await handler.handle_event(event)

        assert result is True
        handler.write_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_calls_mlx_whisper(
        self,
        handler: WhisperEventHandler,
    ) -> None:
        """Test that _transcribe calls mlx_whisper correctly."""
        audio = np.zeros(16000, dtype=np.float32)

        with patch("wyoming_mlx_whisper.handler.mlx_whisper") as mock_mlx:
            mock_mlx.transcribe.return_value = {"text": "test transcription"}
            result = handler._transcribe(audio)

        assert result == "test transcription"
        mock_mlx.transcribe.assert_called_once()
        call_args = mock_mlx.transcribe.call_args
        assert call_args.kwargs["path_or_hf_repo"] == handler._model

    @pytest.mark.asyncio
    async def test_transcribe_with_language(
        self,
        mock_wyoming_info: Info,
        mock_model: str,
    ) -> None:
        """Test that _transcribe passes language when set."""
        handler = WhisperEventHandler(
            mock_wyoming_info,
            mock_model,
            language="en",
            reader=MagicMock(),
            writer=MagicMock(),
        )
        audio = np.zeros(16000, dtype=np.float32)

        with patch("wyoming_mlx_whisper.handler.mlx_whisper") as mock_mlx:
            mock_mlx.transcribe.return_value = {"text": "hello"}
            result = handler._transcribe(audio)

        assert result == "hello"
        call_args = mock_mlx.transcribe.call_args
        assert call_args.kwargs["language"] == "en"

    @pytest.mark.asyncio
    async def test_handle_transcribe_event(self, handler: WhisperEventHandler) -> None:
        """Test handling of Transcribe event."""
        event = Transcribe().event()
        result = await handler.handle_event(event)

        assert result is True

    @pytest.mark.asyncio
    async def test_handle_transcribe_event_with_initial_prompt(
        self,
        handler: WhisperEventHandler,
    ) -> None:
        """Test handling of Transcribe event with initial_prompt in context."""
        event = Transcribe(context={"initial_prompt": "Custom vocabulary"}).event()
        result = await handler.handle_event(event)

        assert result is True
        assert handler._initial_prompt == "Custom vocabulary"

    @pytest.mark.asyncio
    async def test_handle_transcribe_event_without_context(
        self,
        handler: WhisperEventHandler,
    ) -> None:
        """Test handling of Transcribe event without context."""
        event = Transcribe(context=None).event()
        result = await handler.handle_event(event)

        assert result is True
        assert handler._initial_prompt is None

    @pytest.mark.asyncio
    async def test_transcribe_with_initial_prompt(
        self,
        handler: WhisperEventHandler,
    ) -> None:
        """Test that _transcribe passes initial_prompt when set."""
        handler._initial_prompt = "Custom vocabulary hint"
        audio = np.zeros(16000, dtype=np.float32)

        with patch("wyoming_mlx_whisper.handler.mlx_whisper") as mock_mlx:
            mock_mlx.transcribe.return_value = {"text": "transcribed text"}
            result = handler._transcribe(audio)

        assert result == "transcribed text"
        call_args = mock_mlx.transcribe.call_args
        assert call_args.kwargs["initial_prompt"] == "Custom vocabulary hint"

    @pytest.mark.asyncio
    async def test_handle_unknown_event(self, handler: WhisperEventHandler) -> None:
        """Test handling of unknown event type."""
        # Create an event with an unknown type
        event = Event(type="unknown-event-type")
        result = await handler.handle_event(event)

        # Should return True (continue processing)
        assert result is True
