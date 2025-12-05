"""Event handler for clients of the server."""

import argparse
import logging
import time
from typing import Any

import mlx_whisper
import numpy as np
from numpy.typing import NDArray
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

_RATE = 16000
_WIDTH = 2
_CHANNELS = 1


def _pcm_to_float(audio_bytes: bytes) -> NDArray[np.float32]:
    """Convert 16-bit PCM audio bytes to float32 array normalized to [-1, 1]."""
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


class WhisperEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the event handler."""
        super().__init__(*args, **kwargs)
        self._model = cli_args.model
        self._wyoming_info_event = wyoming_info.event()
        self._audio = b""
        self._audio_converter = AudioChunkConverter(
            rate=_RATE,
            width=_WIDTH,
            channels=_CHANNELS,
        )

    def _reset(self) -> None:
        """Reset the audio buffer."""
        self._audio = b""

    def _transcribe(self, audio: NDArray[np.float32]) -> str:
        """Transcribe audio using MLX Whisper."""
        start_time = time.time()
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=self._model)
        elapsed = time.time() - start_time
        _LOGGER.debug("Transcription completed in %.2f seconds", elapsed)
        return str(result["text"])

    async def _handle_audio_chunk(self, event: Event) -> bool:
        """Handle incoming audio chunk."""
        if not self._audio:
            _LOGGER.debug("Receiving audio")
        chunk = AudioChunk.from_event(event)
        chunk = self._audio_converter.convert(chunk)
        self._audio += chunk.audio
        return True

    async def _handle_audio_stop(self) -> bool:
        """Handle end of audio stream and perform transcription."""
        _LOGGER.debug("Audio stopped, starting transcription")
        audio = _pcm_to_float(self._audio)
        text = self._transcribe(audio)
        _LOGGER.info(text)
        await self.write_event(Transcript(text=text).event())
        _LOGGER.debug("Transcription sent")
        self._reset()
        return False

    async def _handle_describe(self) -> bool:
        """Handle describe request."""
        await self.write_event(self._wyoming_info_event)
        _LOGGER.debug("Sent info")
        return True

    async def handle_event(self, event: Event) -> bool:
        """Handle an event from the client."""
        if AudioChunk.is_type(event.type):
            return await self._handle_audio_chunk(event)

        if AudioStop.is_type(event.type):
            return await self._handle_audio_stop()

        if Transcribe.is_type(event.type):
            _LOGGER.debug("Transcribe event")
            return True

        if Describe.is_type(event.type):
            return await self._handle_describe()

        return True
