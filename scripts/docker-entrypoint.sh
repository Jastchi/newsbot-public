#!/bin/bash
# Docker entrypoint for NewsBot API.
# ENABLE_SCHEDULER=true: start API first, wait for healthy, then scheduler.
# Monitoring loop exits if either process dies.

set -e

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
API_URL="http://localhost:${API_PORT}"
ENABLE_SCHEDULER="${ENABLE_SCHEDULER:-false}"
SCHEDULER_PID_FILE="${SCHEDULER_PID_FILE:-/tmp/scheduler.pid}"
# Export so child processes (uvicorn/API) inherit it and can read the PID file path.
export SCHEDULER_PID_FILE

cleanup() {
    echo "Received shutdown signal..."
    [ -n "$SCHEDULER_PID" ] && kill -TERM "$SCHEDULER_PID" 2>/dev/null && wait "$SCHEDULER_PID" 2>/dev/null || true
    [ -n "$SCHEDULER_PID_FILE" ] && rm -f "$SCHEDULER_PID_FILE"
    [ -n "$UVICORN_PID" ] && kill -TERM "$UVICORN_PID" 2>/dev/null && wait "$UVICORN_PID" 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

check_health() {
    python3 -c "
import urllib.request, json, sys
try:
    with urllib.request.urlopen('${API_URL}/health', timeout=5) as r:
        if r.getcode() == 200 and json.loads(r.read().decode()).get('status') == 'ok':
            sys.exit(0)
except Exception:
    pass
sys.exit(1)
" 2>/dev/null
}

if [[ "$ENABLE_SCHEDULER" =~ ^(true|1|yes)$ ]]; then
    echo "Starting API server..."
    uvicorn api.app:app --host "$API_HOST" --port "$API_PORT" &
    UVICORN_PID=$!

    echo "Waiting for API to be healthy..."
    HEALTHY=false
    for i in $(seq 1 60); do
        if check_health; then
            HEALTHY=true
            break
        fi
        if ! kill -0 "$UVICORN_PID" 2>/dev/null; then
            echo "ERROR: API process exited unexpectedly."
            exit 1
        fi
        sleep 2
    done

    if [ "$HEALTHY" = false ]; then
        echo "ERROR: API failed health check."
        kill -TERM "$UVICORN_PID"
        exit 1
    fi

    echo "Starting scheduler service..."
    python scripts/scheduler.py --host localhost --port "$API_PORT" &
    SCHEDULER_PID=$!
    echo "$SCHEDULER_PID" > "$SCHEDULER_PID_FILE"

    # Exits if either process dies
    while true; do
        if ! kill -0 "$UVICORN_PID" 2>/dev/null; then
            echo "API server stopped. Exiting..."
            break
        fi
        if ! kill -0 "$SCHEDULER_PID" 2>/dev/null; then
            echo "Scheduler stopped. Exiting..."
            break
        fi
        sleep 5
    done

    cleanup
else
    echo "Starting API server (Standalone)..."
    exec uvicorn api.app:app --host "$API_HOST" --port "$API_PORT"
fi
