# Deploy: graintrace MCP + Open WebUI (Argo-backed)

Bring up the whole chat stack on the remote box:

```
browser (via ssh -L) ─▶ Open WebUI :8080
                          ├─ OpenAI API ─▶ Argo  https://apps.inside.anl.gov/argoapi/v1   (Claude, key = ANL username)
                          └─ OpenAPI    ─▶ mcpo :8765 ─▶ graintrace-mcp ─▶ NEPER/MOOSE/CUBIT
```

Prereqs (once):
- `graintrace_env` (py3.13) with graintrace + mcpo: `pip install -e ".[mcp]" mcpo`
- `openwebui` env (py3.11): `conda create -n openwebui python=3.11 -y && conda activate openwebui && pip install open-webui`
- Secret file:
  ```bash
  cp deploy/env.example deploy/env
  sed -i "s/change-me.*/$(openssl rand -hex 32)/" deploy/env   # set MCPO_API_KEY
  chmod 600 deploy/env
  ```

The Argo model connection (base URL `https://apps.inside.anl.gov/argoapi/v1`,
API key = your ANL username, model `Claude Opus 4.8`) and the tool server
(`http://localhost:8765`, key = `MCPO_API_KEY`) are configured **once in the Open
WebUI UI** (Settings → Connections / Tools); they persist in Open WebUI's DB, so
they are not part of these service files.

---

## Option A — tmux (simplest; manual start)

```bash
chmod +x deploy/tmux/*.sh
deploy/tmux/start-stack.sh      # starts mcpo + open-webui in a detached session
tmux attach -t graintrace-webui # watch logs (Ctrl-b n to switch windows)
deploy/tmux/stop-stack.sh       # stop everything
```

Survives SSH disconnect (tmux keeps running) but NOT a reboot. For reboot
survival use Option B.

## Option B — systemd --user (survives logout AND reboot)

```bash
# 1. secret file where the units look for it
mkdir -p ~/.config/graintrace-webui && cp deploy/env ~/.config/graintrace-webui/env

# 2. install the units
mkdir -p ~/.config/systemd/user
cp deploy/systemd/graintrace-mcpo.service deploy/systemd/open-webui.service ~/.config/systemd/user/

# 3. let user services run without an active login (key for a remote box)
loginctl enable-linger "$USER"

# 4. enable + start
systemctl --user daemon-reload
systemctl --user enable --now graintrace-mcpo.service open-webui.service

# status / logs
systemctl --user status graintrace-mcpo.service
journalctl --user -u graintrace-mcpo.service -f
journalctl --user -u open-webui.service -f
```

Stop / restart:
```bash
systemctl --user restart open-webui.service
systemctl --user stop graintrace-mcpo.service open-webui.service
systemctl --user disable graintrace-mcpo.service open-webui.service
```

If you edit a `.service` file, re-copy it and `systemctl --user daemon-reload`.

---

## Reach it from your laptop

```bash
ssh -L 8080:localhost:8080 tranh@mom-04.egs.anl.gov
# then open http://localhost:8080
```

## Notes / gotchas

- **Paths:** the units assume conda at `~/miniconda3/envs/{graintrace_env,openwebui}`
  and NEPER at `~/.local/bin`. Edit the `Environment=PATH=` / `ExecStart=` lines
  if yours differ.
- **Argo is network-gated:** it only answers from the ANL network, which is fine
  because Open WebUI runs on the ANL box. From a laptop you'd need ANL VPN.
- **Open WebUI first start** may try to fetch a local embedding model (for RAG).
  If startup stalls with no internet to HuggingFace, either point Open WebUI's
  embedding engine at Argo (Settings → Documents → OpenAI, same connection) or
  set `Environment=HF_HUB_OFFLINE=1` in `open-webui.service` and skip RAG.
- **Secrets:** `deploy/env` and `~/.config/graintrace-webui/env` hold the mcpo
  key — keep them `chmod 600`, never commit (`deploy/env` is gitignored).
