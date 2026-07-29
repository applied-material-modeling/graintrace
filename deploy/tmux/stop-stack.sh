#!/usr/bin/env bash
# Stop the graintrace + Open WebUI tmux stack.
set -euo pipefail
SESSION="graintrace-webui"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
  echo "Stopped tmux session '$SESSION'."
else
  echo "No tmux session '$SESSION' running."
fi
