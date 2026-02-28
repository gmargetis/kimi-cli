# Kimi CLI 🤖

Agentic coding assistant powered by **Kimi K2** via NVIDIA NIM API.  
Works like Claude Code — reads files, writes code, runs commands, connects via SSH, and much more.

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

# Named sessions (saved in ~/.kimi_sessions/)
python3 kimi.py --session myproject "continue working on the auth module"

# Send an image (vision)
python3 kimi.py --image screenshot.png "what's wrong with this UI?"
python3 kimi.py -i https://example.com/diagram.png "explain this architecture"

# Skip task planning
python3 kimi.py --no-plan "build a REST API"
```

## Models

| Alias  | Model                              | Best for             |
|--------|------------------------------------|----------------------|
| fast   | moonshotai/kimi-k2-instruct        | Most tasks (default) |
| smart  | moonshotai/kimi-k2.5               | Complex analysis     |
| think  | moonshotai/kimi-k2-thinking        | Hard problems        |
| latest | moonshotai/kimi-k2-instruct-0905   | Latest version       |

## Features

### Core
- 📁 **File tools** — read, write, edit (with diff preview), list, search
- 🖥️ **Shell** — run commands locally or via SSH
- 🔗 **SSH** — connect to remote hosts, upload/download files
- 🌊 **Streaming** — real-time token output
- 💾 **History** — persistent context across sessions (`--resume`)
- 🧠 **Project context** — auto-reads README/package.json for context
- 💬 **Token tracking** — shows usage and estimated cost after each turn

### New in v2
- 🔀 **Git integration** — run git commands (status, diff, log, commit, push, etc.) as a tool
- 🌐 **Web fetch** — fetch any URL and get clean readable text (HTML stripped)
- 🖼️ **Image input** — `--image` / `-i` flag for vision tasks (local file or URL)
- 🔍 **Project type detection** — auto-detects Node, Python, Rust, Go, Java, PHP
- 📋 **Task planner** — auto-generates a plan for complex tasks before execution (`--no-plan` to skip)
- ↩️ **Undo** — type `undo` in interactive mode to revert the last file change (20-step history)
- 🐳 **Docker tool** — run docker commands locally or on remote hosts
- 🗄️ **Database tool** — execute SQL on SQLite (stdlib) or Postgres/MySQL (with install hints)
- 🔐 **Env manager** — read/write `.env` files with automatic secret masking
- 💾 **Markdown export** — type `save` in interactive mode to export conversation
- 💰 **Cost estimation** — per-turn and session cost based on Kimi K2 NIM pricing
- 📁 **Named sessions** — `--session myname` to save/load named sessions; type `sessions` to list
- 🔄 **Multi-file glob edit** — replace text across all files matching a glob pattern

## Interactive Commands

In interactive mode (`python3 kimi.py`):

| Command    | Action                                       |
|------------|----------------------------------------------|
| `clear`    | Reset context and clear session history      |
| `undo`     | Revert the last file change (up to 20 steps) |
| `save`     | Export conversation to markdown file         |
| `sessions` | List all saved named sessions                |
| `exit`     | Exit the CLI                                 |

## SSH Support

Built-in aliases for quick remote access:

```python
SSH_HOSTS = {
    "mac":     "dma@192.168.1.123",
    "windows": "john@192.168.1.126",
    "pi":      "gmargetis@192.168.1.120",
}
```

Edit `kimi.py` to add your own hosts.

```bash
python3 kimi.py "run npm build on mac and show me the output"
python3 kimi.py "read the error log at /var/log/app.log on the pi"
```

## Tool Reference

| Tool              | Description                                      |
|-------------------|--------------------------------------------------|
| `read_file`       | Read file contents (with optional line range)    |
| `write_file`      | Write/overwrite a file (with undo support)       |
| `edit_file`       | Replace text in a file (with diff preview + undo)|
| `edit_files_glob` | Replace text in all files matching a glob        |
| `run_command`     | Run a shell command                              |
| `list_files`      | List files in a directory                        |
| `search_in_files` | Search text/regex across files (grep)            |
| `git_command`     | Run git commands                                 |
| `fetch_url`       | Fetch a URL and get clean text                   |
| `docker_run`      | Run docker commands (local or remote via SSH)    |
| `db_query`        | Execute SQL (SQLite built-in, Postgres/MySQL with extra deps) |
| `read_env`        | Read `.env` file (secrets auto-masked)           |
| `write_env`       | Write/update `.env` file                         |
| `ssh_run`         | Run command on remote host                       |
| `ssh_upload`      | Upload file via SCP                              |
| `ssh_download`    | Download file via SCP                            |
| `ssh_read_file`   | Read remote file                                 |
| `ssh_write_file`  | Write remote file                                |

## Cost Estimation

Kimi K2 pricing via NVIDIA NIM:
- Input: ~$0.015 / 1k tokens
- Output: ~$0.060 / 1k tokens

Cost is shown after each turn and as a session summary on exit.
