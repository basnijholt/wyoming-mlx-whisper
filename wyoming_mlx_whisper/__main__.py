#!/usr/bin/env python3
"""Wyoming server for MLX Whisper."""

import argparse
import asyncio
import logging
from functools import partial

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .const import WHISPER_LANGUAGES
from .handler import WhisperEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Run the main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-turbo",
        help="MLX whisper model to use",
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format",
        default=logging.BASIC_FORMAT,
        help="Format for log messages",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format=args.log_format,
    )
    _LOGGER.debug(args)

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
                        name=args.model,
                        description=args.model,
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

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    await server.run(partial(WhisperEventHandler, wyoming_info, args))


def run() -> None:
    """Run the Wyoming MLX Whisper server."""
    asyncio.run(main(), debug=True)


if __name__ == "__main__":
    import contextlib

    with contextlib.suppress(KeyboardInterrupt):
        run()
