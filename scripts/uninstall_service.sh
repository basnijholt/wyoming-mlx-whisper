#!/bin/bash
set -e

PLIST_NAME="com.wyoming_mlx_whisper.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"

echo "Uninstalling Wyoming MLX Whisper service..."

if [ -f "$PLIST_DST" ]; then
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    rm "$PLIST_DST"
    echo "Service uninstalled."
else
    echo "Service not installed."
fi
