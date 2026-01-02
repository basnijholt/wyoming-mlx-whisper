#!/usr/bin/env python3
"""Wyoming server for MLX Whisper."""

import logging
from typing import Annotated

import typer

from . import __version__

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
def main(
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
    from rich.logging import RichHandler

    from .server import run_server

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=debug)],
    )
    _LOGGER.debug(
        "model=%s, uri=%s, language=%s, debug=%s",
        model,
        uri,
        language,
        debug,
    )

    run_server(uri, model, language, debug=debug)


def run() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    run()
