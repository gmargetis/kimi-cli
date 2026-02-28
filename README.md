# Kimi CLI 🤖

Agentic coding assistant powered by **Kimi K2** via NVIDIA NIM API.  
Works like Claude Code — reads files, writes code, runs commands, connects via SSH.

## Quick Install

```bash
# One-liner install
curl -fsSL https://raw.githubusercontent.com/gmargetis/kimi-cli/main/install.sh | bash
```

Or manually:

```bash
git clone https://github.com/gmargetis/kimi-cli.git
cd kimi-cli
pip install -r requirements.txt
```

## Setup

Set your NVIDIA NIM API key:

```bash
export NVIDIA_API_KEY="nvapi-..."
# Or add to ~/.bashrc / ~/.zshrc
```

Get a free key at: https://build.nvidia.com

## Usage

```bash
# One-shot task
python3 kimi.py "add error handling to app.py"

# Interactive mode in a project directory
python3 kimi.py -w ~/myproject

# Choose model
python3 kimi.py -m think "refactor this codebase"  # with reasoning
python3 kimi.py -m smart "complex task"             # kimi-k2.5
python3 kimi.py -m fast  "quick fix"                # kimi-k2-instruct (default)

# Resume last session
python3 kimi.py --resume
```

## Models

| Alias  | Model                              | Best for             |
|--------|------------------------------------|----------------------|
| fast   | moonshotai/kimi-k2-instruct        | Most tasks (default) |
| smart  | moonshotai/kimi-k2.5               | Complex analysis     |
| think  | moonshotai/kimi-k2-thinking        | Hard problems        |
| latest | moonshotai/kimi-k2-instruct-0905   | Latest version       |

## Features

- 📁 **File tools** — read, write, edit (with diff preview), list, search
- 🖥️ **Shell** — run commands locally or via SSH
- 🔗 **SSH** — connect to remote hosts, upload/download files
- 🌊 **Streaming** — real-time token output
- 💾 **History** — persistent context across sessions (`--resume`)
- 🧠 **Project context** — auto-reads README/package.json for context
- 💬 **Token tracking** — shows usage after each turn

## SSH Support

Built-in aliases for quick remote access:

```python
SSH_HOSTS = {
    "mac":     "user@192.168.1.123",
    "windows": "user@192.168.1.126",
    "pi":      "user@192.168.1.120",
}
```

Edit `kimi.py` to add your own hosts.

```bash
python3 kimi.py "run npm build on mac and show me the output"
python3 kimi.py "read the error log at /var/log/app.log on the pi"
```
