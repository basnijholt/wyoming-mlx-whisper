"""Tests for the __main__ module."""

from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from wyoming_mlx_whisper.__main__ import (
    DEFAULT_MODEL,
    DEFAULT_URI,
    app,
    version_callback,
)


class TestVersionCallback:
    """Tests for version_callback function."""

    def test_prints_version_and_exits(self) -> None:
        """Test that version callback prints version and exits."""
        with pytest.raises(typer.Exit):
            version_callback(value=True)

    def test_does_nothing_when_false(self) -> None:
        """Test that callback does nothing when value is False."""
        # Should not raise - just returns None
        version_callback(value=False)


class TestCLI:
    """Tests for CLI commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI test runner."""
        return CliRunner()

    def test_help_option(self, runner: CliRunner) -> None:
        """Test that --help works."""
        result = runner.invoke(app, ["--help"], color=False)
        assert result.exit_code == 0
        assert "Wyoming MLX Whisper" in result.stdout
        assert "--uri" in result.stdout
        assert "--model" in result.stdout

    def test_version_option(self, runner: CliRunner) -> None:
        """Test that --version works."""
        from wyoming_mlx_whisper import __version__

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_default_values(self) -> None:
        """Test that default values are correct."""
        assert DEFAULT_URI == "tcp://0.0.0.0:10300"
        assert DEFAULT_MODEL == "mlx-community/whisper-large-v3-turbo"

    def test_main_calls_run_server(self, runner: CliRunner) -> None:
        """Test that main() calls run_server with correct arguments."""
        with patch("wyoming_mlx_whisper.server.run_server") as mock_run:
            runner.invoke(
                app,
                [
                    "--uri",
                    "tcp://localhost:9999",
                    "--model",
                    "test-model",
                    "--language",
                    "en",
                ],
            )

            # run_server should have been called
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0] == "tcp://localhost:9999"  # uri
            assert call_args[0][1] == "test-model"  # model
            assert call_args[0][2] == "en"  # language
            assert call_args[1]["debug"] is False

    def test_main_with_debug(self, runner: CliRunner) -> None:
        """Test that debug flag is passed correctly."""
        with patch("wyoming_mlx_whisper.server.run_server") as mock_run:
            runner.invoke(app, ["--debug"])

            mock_run.assert_called_once()
            assert mock_run.call_args[1]["debug"] is True

    def test_main_with_env_vars(self, runner: CliRunner) -> None:
        """Test that environment variables are respected."""
        with patch("wyoming_mlx_whisper.server.run_server") as mock_run:
            runner.invoke(
                app,
                [],
                env={
                    "WHISPER_URI": "tcp://env-host:8888",
                    "WHISPER_MODEL": "env-model",
                    "WHISPER_LANGUAGE": "fr",
                },
            )

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0] == "tcp://env-host:8888"
            assert call_args[0][1] == "env-model"
            assert call_args[0][2] == "fr"
