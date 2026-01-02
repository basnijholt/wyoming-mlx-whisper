#!/bin/bash
set -e

PLIST_NAME="com.wyoming_mlx_whisper.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"
LOG_DIR="$HOME/Library/Logs/wyoming-mlx-whisper"
PLIST_URL="https://raw.githubusercontent.com/basnijholt/wyoming-mlx-whisper/main/scripts/$PLIST_NAME"

echo "Installing Wyoming MLX Whisper service..."

# Find uv path
UV_PATH=$(which uv 2>/dev/null || echo "")
if [ -z "$UV_PATH" ]; then
    echo "Error: uv not found. Install it from https://docs.astral.sh/uv/"
    exit 1
fi
echo "Using uv at: $UV_PATH"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$HOME/Library/LaunchAgents"

# Download plist and replace placeholders
curl -fsSL "$PLIST_URL" | \
    sed -e "s|<UV-PATH>|$UV_PATH|g" \
        -e "s|<HOME-DIR>|$HOME|g" \
        -e "s|<LOG-DIR>|$LOG_DIR|g" \
    > "$PLIST_DST"

# Unload if already loaded (ignore errors if not loaded)
launchctl bootout gui/$UID "$PLIST_DST" 2>/dev/null || true

# Load the service
launchctl bootstrap gui/$UID "$PLIST_DST"

echo "Service installed and started."
echo "Logs: $LOG_DIR/"
echo ""
echo "To uninstall, run:"
echo "  curl -fsSL https://raw.githubusercontent.com/basnijholt/wyoming-mlx-whisper/main/scripts/uninstall_service.sh | bash"
