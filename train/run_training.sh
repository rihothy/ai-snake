#!/usr/bin/env bash
# Auto-start the web server (if not running) and keep the training loop alive.
# Resumes automatically from the latest checkpoint (train_state.json).
set -u
cd "$(dirname "$0")"

if ! curl -s -o /dev/null http://127.0.0.1:8899/ ; then
    setsid nohup python3 -m http.server 8899 --bind 0.0.0.0 \
        --directory /home/oli/projects/test/ai-snake \
        > /tmp/http_server.log 2>&1 < /dev/null &
fi

export LD_LIBRARY_PATH=$(find ../.venv/lib/python3.12/site-packages/nvidia \
    -maxdepth 2 -name lib -type d | tr '\n' ':')
export TF_CPP_MIN_LOG_LEVEL=1

while true; do
    ../.venv/bin/python main.py >> training.log 2>&1
    echo "[supervisor] main.py exited with code $? at $(date), restarting in 10s" >> training.log
    sleep 10
done
