#!/usr/bin/env python3
"""Wyoming server for MLX Whisper."""

import asyncio
import contextlib
import logging
from typing import Annotated

import typer
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .const import WHISPER_LANGUAGES
from .handler import WhisperEventHandler

_LOGGER = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def version_callback(value: bool) -> None:  # noqa: FBT001
    """Print version and exit."""
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.command()
def main(
    uri: Annotated[str, typer.Option(help="unix:// or tcp://")],
    model: Annotated[
        str,
        typer.Option(help="Name of MLX Whisper model to use"),
    ] = "mlx-community/whisper-large-v3-turbo",
    debug: Annotated[bool, typer.Option(help="Log DEBUG messages")] = False,  # noqa: FBT002
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
    _LOGGER.debug("model=%s, uri=%s, debug=%s", model, uri, debug)

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
