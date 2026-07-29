#!/usr/bin/env bash
# Start the graintrace + Open WebUI stack in a detached tmux session.
#   deploy/tmux/start-stack.sh
# Attach later with:  tmux attach -t graintrace-webui
set -euo pipefail

SESSION="graintrace-webui"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # deploy/

# --- config (override via env or deploy/env) ---------------------------------
[ -f "$HERE/env" ] && set -a && . "$HERE/env" && set +a
: "${MCPO_API_KEY:?set MCPO_API_KEY (copy deploy/env.example -> deploy/env)}"

GRAINTRACE_ENV="${GRAINTRACE_ENV:-$HOME/miniconda3/envs/graintrace_env}"
OPENWEBUI_ENV="${OPENWEBUI_ENV:-$HOME/miniconda3/envs/openwebui}"
MCPO_HOST="${MCPO_HOST:-127.0.0.1}";       MCPO_PORT="${MCPO_PORT:-8765}"
OPENWEBUI_HOST="${OPENWEBUI_HOST:-127.0.0.1}"; OPENWEBUI_PORT="${OPENWEBUI_PORT:-8080}"
export GRAINTRACE_MCP_WORKDIR="${GRAINTRACE_MCP_WORKDIR:-$HOME/graintrace_mcp_out}"
# NEPER / built binaries must be visible to the graintrace-mcp subprocess.
export PATH="$GRAINTRACE_ENV/bin:$HOME/.local/bin:$PATH"
# The env's newer libstdc++ (CXXABI) for neml2 AOTI / neml2-compile / puma-opt (CPFE).
export LD_LIBRARY_PATH="$GRAINTRACE_ENV/lib:${LD_LIBRARY_PATH:-}"

MCPO="$GRAINTRACE_ENV/bin/mcpo"
GRAINTRACE_MCP="$GRAINTRACE_ENV/bin/graintrace-mcp"
OPEN_WEBUI="$OPENWEBUI_ENV/bin/open-webui"
for bin in "$MCPO" "$GRAINTRACE_MCP" "$OPEN_WEBUI"; do
  [ -x "$bin" ] || { echo "ERROR: not found/executable: $bin" >&2; exit 1; }
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already running. Attach: tmux attach -t $SESSION"; exit 0
fi

# window 1: mcpo -> graintrace-mcp
tmux new-session -d -s "$SESSION" -n mcpo
tmux send-keys -t "$SESSION:mcpo" \
  "export GRAINTRACE_MCP_WORKDIR='$GRAINTRACE_MCP_WORKDIR'; export PATH='$PATH'; \
   export LD_LIBRARY_PATH='$LD_LIBRARY_PATH'; \
   '$MCPO' --host $MCPO_HOST --port $MCPO_PORT --api-key '$MCPO_API_KEY' -- '$GRAINTRACE_MCP'" C-m

# window 2: Open WebUI
tmux new-window -t "$SESSION" -n openwebui
tmux send-keys -t "$SESSION:openwebui" \
  "'$OPEN_WEBUI' serve --host $OPENWEBUI_HOST --port $OPENWEBUI_PORT" C-m

echo "Started tmux session '$SESSION':"
echo "  - mcpo       -> http://$MCPO_HOST:$MCPO_PORT/docs"
echo "  - open-webui -> http://$OPENWEBUI_HOST:$OPENWEBUI_PORT"
echo "Attach: tmux attach -t $SESSION   |   Stop: deploy/tmux/stop-stack.sh"
