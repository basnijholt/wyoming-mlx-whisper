"""Wyoming server implementation."""

import asyncio
import contextlib
import logging

from mlx_whisper.load_models import load_model
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .const import WHISPER_LANGUAGES
from .handler import WhisperEventHandler

_LOGGER = logging.getLogger(__name__)


def _create_wyoming_info(model: str) -> Info:
    """Create Wyoming service info."""
    return Info(
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


def run_server(
    uri: str,
    model: str,
    language: str | None,
    *,
    debug: bool,
) -> None:
    """Run the Wyoming MLX Whisper server."""
    _LOGGER.info("🎤 Wyoming MLX Whisper")
    _LOGGER.info("   URI:      %s", uri)
    _LOGGER.info("   Model:    %s", model)
    _LOGGER.info("   Language: %s", language or "auto")

    _LOGGER.info("📦 Loading model...")
    load_model(model)
    _LOGGER.info("✅ Model loaded!")

    wyoming_info = _create_wyoming_info(model)

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
