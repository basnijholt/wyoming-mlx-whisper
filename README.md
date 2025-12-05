# Wyoming MLX Whisper

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for [mlx-whisper](https://pypi.org/project/mlx-whisper) speech-to-text on Apple Silicon.

Uses `mlx-community/whisper-large-v3-turbo` by default, which runs near real-time on M1 Pro and newer.

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10-3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

### Using uv (recommended)

```sh
git clone https://github.com/basnijholt/wyoming-mlx-whisper.git
cd wyoming-mlx-whisper
uv sync
```

### Using pip

```sh
git clone https://github.com/basnijholt/wyoming-mlx-whisper.git
cd wyoming-mlx-whisper
pip install .
```

## Usage

### Run directly

```sh
# With uv
uv run wyoming-mlx-whisper --uri tcp://0.0.0.0:7891

# Or if installed with pip
wyoming-mlx-whisper --uri tcp://0.0.0.0:7891
```

### Run as macOS service (launchd)

Install the service (starts automatically on login):

```sh
./scripts/install_service.sh
```

The server runs at `tcp://localhost:7891` by default.

Uninstall the service:

```sh
./scripts/uninstall_service.sh
```

View logs:

```sh
tail -f log/run.out log/run.err
```

## Options

```
--uri URI          Wyoming server URI (required), e.g., tcp://0.0.0.0:7891
--model MODEL      MLX Whisper model (default: mlx-community/whisper-large-v3-turbo)
--debug            Enable debug logging
```

## Development

```sh
uv sync
uv run pre-commit install
```

## Acknowledgements

- Forked from [vincent861223/wyoming-mlx-whisper](https://github.com/vincent861223/wyoming-mlx-whisper) by Vincent Lin
- Based on [wyoming-whisper-api-client](https://github.com/ser/wyoming-whisper-api-client) by Dr. Serge Victor
