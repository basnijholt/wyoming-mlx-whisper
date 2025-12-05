#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.wyoming_mlx_whisper.plist"
PLIST_SRC="$SCRIPT_DIR/$PLIST_NAME"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"

echo "Installing Wyoming MLX Whisper service..."

# Create log directory
mkdir -p "$PROJECT_DIR/log"

# Copy plist and replace placeholder
sed "s|<PROJECT-DIR>|$PROJECT_DIR|g" "$PLIST_SRC" > "$PLIST_DST"

# Load the service
launchctl load "$PLIST_DST"

echo "Service installed and started."
echo "Logs: $PROJECT_DIR/log/"
