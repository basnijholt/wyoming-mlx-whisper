#!/usr/bin/env python3
"""Wyoming server for MLX Whisper."""

import asyncio
import contextlib
import logging
from typing import Annotated

import typer
from mlx_whisper.load_models import load_model
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .const import WHISPER_LANGUAGES
from .handler import WhisperEventHandler

_LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def version_callback(value: bool) -> None:  # noqa: FBT001
    """Print version and exit."""
    if value:
        typer.echo(__version__)
        raise typer.Exit


DEFAULT_URI = "tcp://0.0.0.0:10300"
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


@app.command()
def main(  # noqa: PLR0913
    uri: Annotated[
        str,
        typer.Option(envvar="WHISPER_URI", help="unix:// or tcp://"),
    ] = DEFAULT_URI,
    model: Annotated[
        str,
        typer.Option(envvar="WHISPER_MODEL", help="Name of MLX Whisper model to use"),
    ] = DEFAULT_MODEL,
    language: Annotated[
        str | None,
        typer.Option(envvar="WHISPER_LANGUAGE", help="Language code (e.g., 'en')"),
    ] = None,
    debug: Annotated[  # noqa: FBT002
        bool,
        typer.Option(envvar="WHISPER_DEBUG", help="Log DEBUG messages"),
    ] = False,
    log_format: Annotated[
        str,
        typer.Option(help="Format for log messages"),
    ] = logging.BASIC_FORMAT,
    version: Annotated[  # noqa: ARG001, FBT002
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Print version and exit",
        ),
    ] = False,
) -> None:
    """Run the Wyoming MLX Whisper server."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=log_format,
    )
    _LOGGER.debug(
        "model=%s, uri=%s, language=%s, debug=%s",
        model,
        uri,
        language,
        debug,
    )

    typer.echo(typer.style("🎤 Wyoming MLX Whisper", fg=typer.colors.GREEN, bold=True))
    typer.echo(f"   URI:      {typer.style(uri, fg=typer.colors.CYAN)}")
    typer.echo(f"   Model:    {typer.style(model, fg=typer.colors.CYAN)}")
    typer.echo(f"   Language: {typer.style(language or 'auto', fg=typer.colors.CYAN)}")

    # Pre-load model to avoid delay on first request
    typer.echo(typer.style("📦 Loading model...", fg=typer.colors.YELLOW))
    load_model(model)
    typer.echo(typer.style("✅ Model loaded!", fg=typer.colors.GREEN))

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="mlx-whisper",
                description="MLX Whisper speech-to-text for Apple Silicon",
                attribution=Attribution(
                    name="MLX Community",
                    url="https://github.com/ml-explore/mlx-examples",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model,
                        description=model,
                        attribution=Attribution(
                            name="OpenAI Whisper",
                            url="https://github.com/openai/whisper",
                        ),
                        installed=True,
                        languages=WHISPER_LANGUAGES,
                        version=__version__,
                    ),
                ],
            ),
        ],
    )

    async def _run() -> None:
        server = AsyncServer.from_uri(uri)
        _LOGGER.info("Ready")
        await server.run(
            lambda *args, **kwargs: WhisperEventHandler(
                wyoming_info,
                model,
                language,
                *args,
                **kwargs,
            ),
        )

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run(), debug=debug)


def run() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    run()
