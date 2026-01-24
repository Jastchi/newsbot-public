#!/bin/bash
# Docker entrypoint script for NewsBot API container.
#
# This script optionally starts the scheduler service alongside the API server
# when the ENABLE_SCHEDULER environment variable is set to true.
#
# Usage:
#   ENABLE_SCHEDULER=true  - Start scheduler in background, then API server
#   ENABLE_SCHEDULER=false - Start API server only (default)

set -e

# Function to handle shutdown signals
cleanup() {
    echo "Received shutdown signal..."
    if [ -n "$SCHEDULER_PID" ]; then
        echo "Stopping scheduler (PID: $SCHEDULER_PID)..."
        kill -TERM "$SCHEDULER_PID" 2>/dev/null || true
        wait "$SCHEDULER_PID" 2>/dev/null || true
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Check if scheduler should be enabled
ENABLE_SCHEDULER="${ENABLE_SCHEDULER:-false}"

if [ "$ENABLE_SCHEDULER" = "true" ] || [ "$ENABLE_SCHEDULER" = "1" ] || [ "$ENABLE_SCHEDULER" = "yes" ]; then
    echo "Starting scheduler service..."
    
    # Start scheduler in background, connecting to localhost API
    python scripts/scheduler.py --host localhost --port 8000 &
    SCHEDULER_PID=$!
    
    # Export PID to environment variables
    export SCHEDULER_PID=$SCHEDULER_PID
    
    echo "Scheduler started with PID: $SCHEDULER_PID"
    
    # Give scheduler a moment to initialize
    sleep 2
    
    # Check if scheduler is still running
    if ! kill -0 "$SCHEDULER_PID" 2>/dev/null; then
        echo "WARNING: Scheduler process may have exited unexpectedly"
    fi
else
    echo "Scheduler disabled (set ENABLE_SCHEDULER=true to enable)"
fi

echo "Starting API server..."

# Start API server in foreground
# Using exec to replace this script process with uvicorn
exec uvicorn api.app:app --host 0.0.0.0 --port 8000
