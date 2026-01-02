"""Tests for the server module."""

from unittest.mock import patch

from wyoming_mlx_whisper import __version__
from wyoming_mlx_whisper.const import WHISPER_LANGUAGES
from wyoming_mlx_whisper.server import _create_wyoming_info, run_server


class TestCreateWyomingInfo:
    """Tests for _create_wyoming_info function."""

    def test_creates_info_with_model(self) -> None:
        """Test that info is created with the correct model."""
        model = "mlx-community/whisper-large-v3-turbo"
        info = _create_wyoming_info(model)

        assert info.asr is not None
        assert len(info.asr) == 1
        assert info.asr[0].name == "mlx-whisper"
        assert len(info.asr[0].models) == 1
        assert info.asr[0].models[0].name == model

    def test_includes_whisper_languages(self) -> None:
        """Test that all Whisper languages are included."""
        info = _create_wyoming_info("test-model")

        assert info.asr[0].models[0].languages == WHISPER_LANGUAGES

    def test_includes_version(self) -> None:
        """Test that version is included in info."""
        info = _create_wyoming_info("test-model")

        assert info.asr[0].version == __version__
        assert info.asr[0].models[0].version == __version__

    def test_attribution(self) -> None:
        """Test that attribution is set correctly."""
        info = _create_wyoming_info("test-model")

        # Program attribution
        assert info.asr[0].attribution.name == "MLX Community"
        assert "ml-explore" in info.asr[0].attribution.url

        # Model attribution
        assert info.asr[0].models[0].attribution.name == "OpenAI Whisper"
        assert "openai" in info.asr[0].models[0].attribution.url

    def test_installed_flags(self) -> None:
        """Test that installed flags are set."""
        info = _create_wyoming_info("test-model")

        assert info.asr[0].installed is True
        assert info.asr[0].models[0].installed is True


class TestRunServer:
    """Tests for run_server function."""

    def test_logs_startup_banner(self) -> None:
        """Test that run_server logs startup information."""
        with (
            patch("wyoming_mlx_whisper.server._LOGGER") as mock_logger,
            patch("wyoming_mlx_whisper.server.load_model"),
            patch("wyoming_mlx_whisper.server.asyncio.run"),
        ):
            run_server(
                uri="tcp://localhost:10300",
                model="test-model",
                language="en",
                debug=False,
            )

            # Check that startup messages were logged
            calls = " ".join(str(call) for call in mock_logger.info.call_args_list)
            assert "Wyoming MLX Whisper" in calls
            assert "Loading model" in calls
            assert "Model loaded" in calls

    def test_loads_model(self) -> None:
        """Test that run_server loads the specified model."""
        with (
            patch("wyoming_mlx_whisper.server._LOGGER"),
            patch("wyoming_mlx_whisper.server.load_model") as mock_load,
            patch("wyoming_mlx_whisper.server.asyncio.run"),
        ):
            run_server(
                uri="tcp://localhost:10300",
                model="mlx-community/whisper-tiny",
                language=None,
                debug=False,
            )

            mock_load.assert_called_once_with("mlx-community/whisper-tiny")

    def test_runs_async_server(self) -> None:
        """Test that run_server starts the async server."""
        with (
            patch("wyoming_mlx_whisper.server._LOGGER"),
            patch("wyoming_mlx_whisper.server.load_model"),
            patch("wyoming_mlx_whisper.server.asyncio.run") as mock_run,
        ):
            run_server(
                uri="tcp://localhost:10300",
                model="test-model",
                language=None,
                debug=True,
            )

            mock_run.assert_called_once()
            # Check debug flag was passed
            assert mock_run.call_args[1]["debug"] is True

    def test_handles_keyboard_interrupt(self) -> None:
        """Test that KeyboardInterrupt is handled gracefully."""
        with (
            patch("wyoming_mlx_whisper.server._LOGGER"),
            patch("wyoming_mlx_whisper.server.load_model"),
            patch(
                "wyoming_mlx_whisper.server.asyncio.run",
                side_effect=KeyboardInterrupt,
            ),
        ):
            # Should not raise
            run_server(
                uri="tcp://localhost:10300",
                model="test-model",
                language=None,
                debug=False,
            )

    def test_logs_language_auto_when_none(self) -> None:
        """Test that 'auto' is logged when language is None."""
        with (
            patch("wyoming_mlx_whisper.server._LOGGER") as mock_logger,
            patch("wyoming_mlx_whisper.server.load_model"),
            patch("wyoming_mlx_whisper.server.asyncio.run"),
        ):
            run_server(
                uri="tcp://localhost:10300",
                model="test-model",
                language=None,
                debug=False,
            )

            # Check that 'auto' was used for language
            calls = " ".join(str(call) for call in mock_logger.info.call_args_list)
            assert "auto" in calls
