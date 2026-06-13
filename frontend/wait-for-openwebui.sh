#!/bin/sh
set -eu

ready_url="${OPENWEBUI_READY_URL:-http://openwebui:8080/ready}"
timeout_seconds="${OPENWEBUI_READY_TIMEOUT:-180}"
deadline="$(($(date +%s) + timeout_seconds))"

while [ "$(date +%s)" -lt "$deadline" ]; do
    if wget -q -T 3 -O - "$ready_url" >/dev/null 2>&1; then
        echo "OpenWebUI is ready"
        exit 0
    fi

    echo "Waiting for OpenWebUI readiness..."
    sleep 2
done

echo "OpenWebUI did not become ready before timeout: $ready_url" >&2
exit 1
