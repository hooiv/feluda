#!/bin/bash
set -e

# Function to handle signals
function handle_signal {
    echo "Received signal, shutting down gracefully..."
    # Add any cleanup logic here
    exit 0
}

# Set up signal handlers
trap handle_signal SIGTERM SIGINT

# Check if we're running as the entrypoint
if [ "$1" = "python" ] && [ "$2" = "-m" ] && [ "$3" = "feluda" ]; then
    echo "Starting Feluda..."
    
    # Check if a config file is provided
    if [ -n "$FELUDA_CONFIG" ] && [ -f "$FELUDA_CONFIG" ]; then
        echo "Using config file: $FELUDA_CONFIG"
        exec python -m feluda --config "$FELUDA_CONFIG" "$@"
    else
        echo "No config file provided, using default configuration"
        exec python -m feluda "$@"
    fi
else
    # If not running as the entrypoint, just execute the command
    exec "$@"
fi
