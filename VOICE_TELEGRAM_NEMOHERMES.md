# Telegram Voice Notes with NeMoHermes and Qwen3.8-Flash-Next

This guide documents the complete working setup for running a local, voice-enabled autonomous agent on Telegram using **NeMoHermes / OpenShell** backed by **Cogni-Brain** (Qwen3.8-Flash-Next NVFP4 + HashK PLE running on NVIDIA DGX Spark).

Telegram voice notes do not go directly to the LLM by default. The runtime pipeline is:

```text
Telegram voice note
-> Hermes Gateway
-> local speech-to-text (faster-whisper)
-> Cogni-Brain (Qwen3.8-Flash-Next)
-> Telegram reply
```

While Qwen3.8-Flash-Next provides fast reasoning and tool execution, the Telegram gateway requires a dedicated speech-to-text layer first. In this setup, local STT is handled entirely on-device with `faster-whisper`.

---

## Final Working Configuration

```text
STT provider: local
STT engine: faster-whisper
Whisper model: Systran/faster-whisper-base
Model location inside sandbox: /sandbox/models/faster-whisper-base-flat
Hermes config: /sandbox/.hermes/config.yaml
Gateway boot script: ~/boot.sh
Backend Inference: http://127.0.0.1:8000/v1 (Model: Cogni-Brain)
```

---

## 1. Important Note: Avoid `nemoclaw cogni rebuild --force`

If a sandbox encounters configuration drift or errors, do **not** run `rebuild --force`. The base image rebuild currently fails due to upstream Debian trixie apt dependency solver conflicts (`libssl-dev` vs `libssl3t64`).

Instead, destroy and recreate the sandbox cleanly while keeping the shared gateway:

```bash
nemohermes cogni destroy
```
* Confirm deletion: `yes`
* When prompted: *"Also destroy the shared NemoClaw gateway?"*, press **Enter** (`N`) to preserve the gateway and keep the next setup fast.

---

## 2. Onboard Sandbox to Local Cogni-Brain Endpoint

Run on the host:

```bash
nemohermes onboard
```

### Configuration Selections:
1. **Inference Provider**: Select `4` (*Other OpenAI-compatible endpoint*)
2. **Base URL**: `http://127.0.0.1:8000/v1`
3. **API Key**: `local-bypass` (or press Enter if unauthenticated)
4. **Model Name**: `Cogni-Brain`
5. **Sandbox Name**: `cogni`
6. **Web Search**: Select `2` (*Tavily Search*)
7. **Messaging Channel**: Toggle `1` (*Telegram*)
   * Enter your Telegram Bot Token from `@BotFather`.
   * Reply mode: `n` (*Reply to all messages, not just @mentions*).
   * Enter your numeric Telegram User ID from `@userinfobot`.
   * Resource Profile: Choose `4` (*developer: 75% CPU / 75% RAM*).
8. **Policy Presets**: Choose `Balanced` defaults.

---

## 3. Install `faster-whisper` Inside the Sandbox

Connect to the sandbox:

```bash
nemohermes cogni connect
```

Install `faster-whisper` using pip into the sandbox user site:

```bash
/usr/bin/pip3 install --break-system-packages --user faster-whisper
```

Verify the installation:

```bash
/usr/bin/python3 -c "import faster_whisper; print('faster-whisper OK')"
```

Expected output:
```text
faster-whisper OK
```

---

## 4. Ensure Hermes Gateway Can See Sandbox User Packages

Hermes runs inside its own virtual environment, so the gateway needs `PYTHONPATH` exported to locate packages installed via `/usr/bin/pip3 --user`.

Create `~/boot.sh` inside the sandbox:

```bash
cat << 'EOF' > ~/boot.sh
#!/bin/bash
# OpenShell sandbox does not load user site packages into Hermes venv

# 1. Expose user site-packages
export PYTHONPATH=/sandbox/.local/lib/python3.13/site-packages:${PYTHONPATH:-}

# 2. Clear ghost locks
rm -f ~/.hermes/gateway.pid
rm -rf ~/.local/state/hermes/gateway-locks/

# 3. Start gateway in background
nohup hermes gateway run --replace > ~/gateway.log 2>&1 &
echo "Local STT path initialized."
EOF

chmod +x ~/boot.sh
```

---

## 5. Enable Local STT in Hermes Config

Append the local STT configuration to `/sandbox/.hermes/config.yaml`:

```yaml
cat << 'EOF' >> /sandbox/.hermes/config.yaml

stt:
  enabled: true
  provider: "local"
  local:
    model: "/sandbox/models/faster-whisper-base-flat"
EOF
```

> [!NOTE]
> Setting `model: "base"` directly in `config.yaml` will cause `faster-whisper` to attempt a download from Hugging Face at runtime. Inside an OpenShell sandbox, that download is blocked by egress security policies, resulting in a `403 Forbidden` error. Using a pre-staged local path avoids runtime network calls.

Exit the sandbox back to the host shell:

```bash
exit
```

---

## 6. Download and Prepare Whisper Model on the Host

On the host machine, download the model snapshot using `uvx`:

```bash
uvx --from huggingface-hub hf download Systran/faster-whisper-base
```

Hugging Face cache snapshots use symlinks. Flatten the files into a clean directory:

```bash
rm -rf /tmp/faster-whisper-base-flat
mkdir -p /tmp/faster-whisper-base-flat

cp -L ~/.cache/huggingface/hub/models--Systran--faster-whisper-base/snapshots/*/* \
  /tmp/faster-whisper-base-flat/

ls -lh /tmp/faster-whisper-base-flat
```

Expected files:
```text
config.json
model.bin (~139M)
README.md
tokenizer.json
vocabulary.txt
```

Package into a tarball:

```bash
cd /tmp
tar -czf faster-whisper-base-flat.tgz faster-whisper-base-flat
```

---

## 7. Upload and Unpack Model in Sandbox

Upload the tarball into the sandbox:

```bash
openshell sandbox upload cogni /tmp/faster-whisper-base-flat.tgz /sandbox/
```

Reconnect to the sandbox:

```bash
nemohermes cogni connect
```

Extract the model files:

```bash
mkdir -p /sandbox/models
tar -xzf /sandbox/faster-whisper-base-flat.tgz -C /sandbox/models
rm /sandbox/faster-whisper-base-flat.tgz
```

Verify the files are in place:

```bash
ls -lh /sandbox/models/faster-whisper-base-flat
```

---

## 8. Launch Gateway and Test

Inside the sandbox, run the boot script:

```bash
./boot.sh
```

Output:
```text
Local STT path initialized.
```

Exit back to host:
```bash
exit
```

Optional: If your agent performs deep thinking or extensive tool calling chains, bump the inference gateway timeout to prevent premature disconnects:

```bash
openshell inference set --provider compatible-endpoint --model Cogni-Brain --timeout 600 --no-verify
```

---

## 9. Failure Signatures and Diagnostics

| Error Signature | Root Cause | Fix |
| :--- | :--- | :--- |
| `no speech-to-text provider is configured` | `stt` block missing from `config.yaml` | Add `stt.enabled: true`, `provider: "local"`, and path to model in `/sandbox/.hermes/config.yaml`. |
| `ModuleNotFoundError: No module named 'faster_whisper'` | Package missing or `PYTHONPATH` not set | Run `/usr/bin/pip3 install --break-system-packages --user faster-whisper` and launch via `~/boot.sh`. |
| `httpx.ProxyError: 403 Forbidden` during STT | Sandbox network policy blocking HF download | Stage model on host and upload flat directory to `/sandbox/models/faster-whisper-base-flat`. |
| Gateway restarts or locks on boot | Stale PID lock files from prior container run | `rm -f ~/.hermes/gateway.pid` and `rm -rf ~/.local/state/hermes/gateway-locks/` (handled in `boot.sh`). |

---

## 10. Working Result

Once configured, Telegram voice notes are transcribed on-device with zero cloud latency or privacy exposure:

1. User sends Telegram voice note: *"Can you hear me?"*
2. Local `faster-whisper` transcribes audio to text inside the sandbox.
3. Qwen3.8-Flash-Next receives prompt and streams reasoning tokens.
4. Telegram delivers text reply: *"Yes, loud and clear. Your messages are coming through fine on Telegram. What do you need?"*
