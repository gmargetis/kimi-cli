#!/usr/bin/env python3
"""
Kimi CLI - Agentic coding assistant powered by Kimi via NVIDIA NIM
Usage:
  python3 kimi.py "task"                  # one-shot
  python3 kimi.py -w /path/to/project     # interactive in project dir
  python3 kimi.py --model kimi-k2-thinking "hard task"
  python3 kimi.py --session myproject     # named session
  python3 kimi.py --image screenshot.png "what's in this image?"
  python3 kimi.py --orchestrate "build auth module"  # multi-agent orchestration
  python3 kimi.py -O "fix all TypeScript errors"     # orchestrate shorthand
  python3 kimi.py --issue 42              # fix GitHub issue #42 and optionally create PR
  python3 kimi.py --issue 42 --pr         # fix issue and auto-create PR
  python3 kimi.py --index                 # build semantic search index for current dir
  python3 kimi.py --search "auth logic"   # semantic search over indexed code
  python3 kimi.py --tui                   # launch rich-based interactive TUI
"""

import os
import sys
import json
import subprocess
import argparse
import signal
import pickle
import sqlite3
import tempfile
import base64
import glob as glob_module
import re
import threading
import concurrent.futures
from pathlib import Path
from datetime import datetime
from urllib import request as urllib_request
from urllib.error import URLError
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.text import Text
from rich.layout import Layout
from rich.align import Align
from rich import print as rprint

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("NVIDIA_API_KEY", "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")
BASE_URL = "https://integrate.api.nvidia.com/v1"
HISTORY_FILE = Path.home() / ".kimi_history.pkl"
SESSIONS_DIR = Path.home() / ".kimi_sessions"

# Known SSH hosts (name → user@host)
SSH_HOSTS = {
    "mac":     "dma@192.168.1.123",
    "macbook": "dma@192.168.1.123",
    "windows": "john@192.168.1.126",
    "pc":      "john@192.168.1.126",
    "pi":      "gmargetis@192.168.1.120",
}

MODELS = {
    "fast":         "moonshotai/kimi-k2-instruct",
    "smart":        "moonshotai/kimi-k2.5",
    "think":        "moonshotai/kimi-k2-thinking",
    "latest":       "moonshotai/kimi-k2-instruct-0905",
    "orchestrator": "moonshotai/kimi-k2.5",
}

# ── File locks for orchestrator (one lock per file path) ──────────────────────
_file_locks: dict = {}
_file_locks_mutex = threading.Lock()

def get_file_lock(path: str) -> threading.Lock:
    """Return (or create) a per-file lock to serialize concurrent writes."""
    with _file_locks_mutex:
        if path not in _file_locks:
            _file_locks[path] = threading.Lock()
        return _file_locks[path]

# Kimi K2 pricing via NVIDIA NIM (per 1k tokens)
COST_PER_1K_INPUT  = 0.015
COST_PER_1K_OUTPUT = 0.060

client  = OpenAI(api_key=API_KEY, base_url=BASE_URL)
console = Console()

# ── Undo stack ────────────────────────────────────────────────────────────────
undo_stack = []  # list of (path, original_content)
MAX_UNDO = 20

def push_undo(path, content):
    undo_stack.append((str(path), content))
    if len(undo_stack) > MAX_UNDO:
        undo_stack.pop(0)

# ── Graceful interrupt ────────────────────────────────────────────────────────
interrupted = False
def _sigint(sig, frame):
    global interrupted
    interrupted = True
    console.print("\n[yellow]⚡ Interrupted — finishing current step...[/yellow]")
signal.signal(signal.SIGINT, _sigint)

# ── Tools definition ──────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "lines_from": {"type": "integer", "description": "Start line (1-indexed, optional)"},
                    "lines_to":   {"type": "integer", "description": "End line (optional)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates or overwrites). Supports undo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace specific text in a file (shows diff before applying). Supports undo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":     {"type": "string"},
                    "old_text": {"type": "string", "description": "Exact text to find"},
                    "new_text": {"type": "string", "description": "Replacement text"}
                },
                "required": ["path", "old_text", "new_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command and return output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "workdir": {"type": "string", "description": "Working directory (optional)"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":      {"type": "string", "description": "Directory (default: .)"},
                    "recursive": {"type": "boolean"},
                    "pattern":   {"type": "string", "description": "Glob pattern (e.g. *.py)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": "Search for text/pattern across files (grep)",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text or regex to search"},
                    "path":    {"type": "string", "description": "Directory to search in (default: .)"},
                    "file_pattern": {"type": "string", "description": "File filter e.g. *.py"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ssh_run",
            "description": "Run a command on a remote host via SSH. Known hosts: mac/macbook (dma@192.168.1.123), windows/pc (john@192.168.1.126), pi (gmargetis@192.168.1.120). Can also use custom user@host.",
            "parameters": {
                "type": "object",
                "properties": {
                    "host":    {"type": "string", "description": "Host alias (mac/windows/pi) or user@host"},
                    "command": {"type": "string", "description": "Shell command to run remotely"}
                },
                "required": ["host", "command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ssh_upload",
            "description": "Upload a local file to a remote host via SCP",
            "parameters": {
                "type": "object",
                "properties": {
                    "host":        {"type": "string", "description": "Host alias or user@host"},
                    "local_path":  {"type": "string", "description": "Local file path"},
                    "remote_path": {"type": "string", "description": "Remote destination path"}
                },
                "required": ["host", "local_path", "remote_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ssh_download",
            "description": "Download a file from a remote host via SCP",
            "parameters": {
                "type": "object",
                "properties": {
                    "host":        {"type": "string", "description": "Host alias or user@host"},
                    "remote_path": {"type": "string", "description": "Remote file path"},
                    "local_path":  {"type": "string", "description": "Local destination path"}
                },
                "required": ["host", "remote_path", "local_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ssh_read_file",
            "description": "Read a file from a remote host",
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "Host alias or user@host"},
                    "path": {"type": "string", "description": "Remote file path"}
                },
                "required": ["host", "path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ssh_write_file",
            "description": "Write a file on a remote host",
            "parameters": {
                "type": "object",
                "properties": {
                    "host":    {"type": "string", "description": "Host alias or user@host"},
                    "path":    {"type": "string", "description": "Remote file path"},
                    "content": {"type": "string", "description": "File content"}
                },
                "required": ["host", "path", "content"]
            }
        }
    },
    # ── New tools ──────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "git_command",
            "description": "Run git commands (status, diff, log, commit, push, branch, checkout, etc.) in the working directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Git subcommand and args, e.g. 'status', 'diff HEAD', 'log --oneline -5', 'commit -m \"msg\"'"},
                    "workdir": {"type": "string", "description": "Working directory (default: cwd)"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a URL and return readable text content (HTML tags stripped)",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "HTTP/HTTPS URL to fetch"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "docker_run",
            "description": "Run docker commands locally or on a remote host via SSH. Commands: build, run, ps, logs, stop, exec, images, pull",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Docker subcommand and args, e.g. 'ps', 'logs mycontainer', 'build -t myapp .'"},
                    "host":    {"type": "string", "description": "Optional SSH host alias or user@host for remote execution"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "db_query",
            "description": "Execute SQL on a SQLite file or Postgres/MySQL via connection string",
            "parameters": {
                "type": "object",
                "properties": {
                    "connection": {"type": "string", "description": "SQLite file path OR connection string like 'postgresql://user:pass@host/db' or 'mysql://user:pass@host/db'"},
                    "sql":        {"type": "string", "description": "SQL query to execute"},
                    "params":     {"type": "array", "items": {"type": "string"}, "description": "Optional query parameters"}
                },
                "required": ["connection", "sql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_env",
            "description": "Read a .env file and return key-value pairs (secret values are masked in logs)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": ".env file path (default: .env)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_env",
            "description": "Write key-value pairs to a .env file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":   {"type": "string", "description": ".env file path (default: .env)"},
                    "values": {"type": "object", "description": "Key-value pairs to write/update"}
                },
                "required": ["values"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_files_glob",
            "description": "Replace text in ALL files matching a glob pattern. Returns list of changed files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern":  {"type": "string", "description": "Glob pattern, e.g. 'src/**/*.py' or '*.js'"},
                    "old_text": {"type": "string", "description": "Exact text to find"},
                    "new_text": {"type": "string", "description": "Replacement text"}
                },
                "required": ["pattern", "old_text", "new_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Semantic search over indexed code files. Returns relevant code chunks by meaning. Run --index first to build the index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query about the code"},
                    "top_k": {"type": "integer", "description": "Number of results to return (default 5)"}
                },
                "required": ["query"]
            }
        }
    }
]

# ── Tool implementations ──────────────────────────────────────────────────────
def read_file(path, lines_from=None, lines_to=None):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        if lines_from or lines_to:
            lines = lines[(lines_from or 1) - 1 : lines_to]
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {e}"

def write_file(path, content):
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Serialize concurrent writes to the same file
        lock = get_file_lock(str(p.resolve()))
        with lock:
            # Save to undo stack if file exists
            if p.exists():
                push_undo(p, p.read_text(encoding="utf-8"))
            p.write_text(content, encoding="utf-8")
        return f"✅ Written {len(content)} chars to {path}"
    except Exception as e:
        return f"❌ Error: {e}"

def edit_file(path, old_text, new_text):
    try:
        p = Path(path)
        lock = get_file_lock(str(p.resolve()))
        with lock:
            content = p.read_text(encoding="utf-8")
            if old_text not in content:
                return f"❌ Text not found in {path}"
            # Save to undo stack
            push_undo(path, content)
            # Show diff
            old_lines = old_text.splitlines()
            new_lines = new_text.splitlines()
            console.print(f"[dim]  📝 Diff in {path}:[/dim]")
            for line in old_lines[:5]:
                console.print(f"[dim red]  - {line}[/dim red]")
            for line in new_lines[:5]:
                console.print(f"[dim green]  + {line}[/dim green]")
            if len(old_lines) > 5:
                console.print(f"[dim]  ... ({len(old_lines)} lines total)[/dim]")
            new_content = content.replace(old_text, new_text, 1)
            p.write_text(new_content, encoding="utf-8")
        return f"✅ Edited {path}"
    except Exception as e:
        return f"❌ Error: {e}"

def run_command(command, workdir=None):
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            cwd=workdir or os.getcwd(), timeout=120
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        combined = out
        if err:
            combined += f"\n[stderr] {err}"
        if result.returncode != 0:
            combined += f"\n[exit code: {result.returncode}]"
        return combined or "(no output)"
    except subprocess.TimeoutExpired:
        return "❌ Command timed out (120s)"
    except Exception as e:
        return f"❌ Error: {e}"

def list_files(path=".", recursive=False, pattern=None):
    try:
        p = Path(path)
        glob = f"**/{pattern or '*'}" if recursive else (pattern or "*")
        files = sorted(str(f.relative_to(p)) for f in p.glob(glob) if f.is_file())
        return "\n".join(files) if files else "(empty)"
    except Exception as e:
        return f"❌ Error: {e}"

def search_in_files(pattern, path=".", file_pattern=None):
    cmd = f"grep -rn --include='{file_pattern or '*'}' {json.dumps(pattern)} {path} 2>/dev/null | head -50"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip() or "(no matches)"

def _resolve_host(host):
    """Resolve host alias to user@host string"""
    return SSH_HOSTS.get(host.lower(), host)

def _ssh_opts():
    return "-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"

def ssh_run(host, command):
    target = _resolve_host(host)
    cmd = f"ssh {_ssh_opts()} {target} {json.dumps(command)}"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        out = result.stdout.strip()
        err = result.stderr.strip()
        combined = out
        if err:
            combined += f"\n[stderr] {err}"
        if result.returncode != 0:
            combined += f"\n[exit: {result.returncode}]"
        return combined or "(no output)"
    except subprocess.TimeoutExpired:
        return "❌ SSH command timed out (60s)"
    except Exception as e:
        return f"❌ SSH error: {e}"

def ssh_upload(host, local_path, remote_path):
    target = _resolve_host(host)
    cmd = f"scp {_ssh_opts()} {local_path} {target}:{remote_path}"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return f"✅ Uploaded {local_path} → {target}:{remote_path}"
        return f"❌ SCP failed: {result.stderr.strip()}"
    except Exception as e:
        return f"❌ SCP error: {e}"

def ssh_download(host, remote_path, local_path):
    target = _resolve_host(host)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = f"scp {_ssh_opts()} {target}:{remote_path} {local_path}"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return f"✅ Downloaded {target}:{remote_path} → {local_path}"
        return f"❌ SCP failed: {result.stderr.strip()}"
    except Exception as e:
        return f"❌ SCP error: {e}"

def ssh_read_file(host, path):
    return ssh_run(host, f"cat {json.dumps(path)}")

def ssh_write_file(host, path, content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', delete=False, encoding='utf-8') as f:
        f.write(content)
        tmp = f.name
    try:
        result = ssh_upload(host, tmp, path)
        return result
    finally:
        os.unlink(tmp)

# ── New tool implementations ──────────────────────────────────────────────────

def git_command(command, workdir=None):
    """Run a git command in the working directory."""
    cwd = workdir or os.getcwd()
    cmd = f"git {command}"
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            cwd=cwd, timeout=60
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        combined = out
        if err:
            combined += f"\n[stderr] {err}"
        if result.returncode != 0:
            combined += f"\n[exit code: {result.returncode}]"
        return combined or "(no output)"
    except subprocess.TimeoutExpired:
        return "❌ Git command timed out (60s)"
    except Exception as e:
        return f"❌ Git error: {e}"

def fetch_url(url):
    """Fetch a URL and return clean text content."""
    try:
        req = urllib_request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; KimiCLI/1.0)"}
        )
        with urllib_request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        # Strip HTML tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&#39;', "'", text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:4000] + ("..." if len(text) > 4000 else "")
    except URLError as e:
        return f"❌ URL error: {e}"
    except Exception as e:
        return f"❌ Fetch error: {e}"

def docker_run(command, host=None):
    """Run a docker command locally or via SSH."""
    full_cmd = f"docker {command}"
    if host:
        return ssh_run(host, full_cmd)
    else:
        return run_command(full_cmd)

def db_query(connection, sql, params=None):
    """Execute SQL on SQLite or Postgres/MySQL."""
    params = params or []
    # Detect SQLite (file path or :memory:)
    if connection == ":memory:" or (
        not connection.startswith("postgresql://") and
        not connection.startswith("postgres://") and
        not connection.startswith("mysql://") and
        not connection.startswith("mysql+") and
        not connection.startswith("sqlite+")
    ):
        # Treat as SQLite file path
        try:
            conn = sqlite3.connect(connection)
            cur = conn.cursor()
            cur.execute(sql, params)
            if cur.description:
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
                conn.close()
                lines = [" | ".join(cols)]
                lines.append("-" * len(lines[0]))
                for row in rows[:100]:
                    lines.append(" | ".join(str(v) for v in row))
                if len(rows) > 100:
                    lines.append(f"... ({len(rows)} rows total, showing 100)")
                return "\n".join(lines)
            else:
                conn.commit()
                affected = cur.rowcount
                conn.close()
                return f"✅ Query OK, {affected} rows affected"
        except Exception as e:
            return f"❌ SQLite error: {e}"
    elif connection.startswith("postgresql://") or connection.startswith("postgres://"):
        return (
            "❌ PostgreSQL requires psycopg2: `pip install psycopg2-binary`\n"
            "Then use: import psycopg2; conn = psycopg2.connect(connection_string)"
        )
    elif connection.startswith("mysql://") or connection.startswith("mysql+"):
        return (
            "❌ MySQL requires pymysql: `pip install pymysql`\n"
            "Then use: import pymysql; conn = pymysql.connect(...)"
        )
    else:
        return f"❌ Unknown connection type: {connection}"

def read_env(path=".env"):
    """Read a .env file and return key-value pairs."""
    p = Path(path)
    if not p.exists():
        return f"❌ File not found: {path}"
    try:
        result = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip()
        # Mask secrets in output
        masked = {}
        secret_keywords = {"key", "secret", "password", "passwd", "token", "api", "auth", "credential"}
        for k, v in result.items():
            if any(kw in k.lower() for kw in secret_keywords):
                masked[k] = "***"
            else:
                masked[k] = v
        return json.dumps(masked, indent=2)
    except Exception as e:
        return f"❌ Error reading .env: {e}"

def write_env(values, path=".env"):
    """Write/update key-value pairs in a .env file."""
    p = Path(path)
    try:
        existing = {}
        lines = []
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key = stripped.split("=", 1)[0].strip()
                    existing[key] = len(lines)
                lines.append(line)
        # Update existing keys or append new ones
        for key, value in values.items():
            if key in existing:
                lines[existing[key]] = f"{key}={value}"
            else:
                lines.append(f"{key}={value}")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        secret_keywords = {"key", "secret", "password", "passwd", "token", "api", "auth", "credential"}
        masked_keys = [k for k in values if any(kw in k.lower() for kw in secret_keywords)]
        if masked_keys:
            return f"✅ Written to {path} (secret keys masked: {', '.join(masked_keys)})"
        return f"✅ Written {len(values)} keys to {path}"
    except Exception as e:
        return f"❌ Error writing .env: {e}"

def edit_files_glob(pattern, old_text, new_text):
    """Replace text in all files matching a glob pattern."""
    matches = glob_module.glob(pattern, recursive=True)
    changed = []
    errors = []
    for filepath in matches:
        p = Path(filepath)
        if not p.is_file():
            continue
        try:
            lock = get_file_lock(str(p.resolve()))
            with lock:
                content = p.read_text(encoding="utf-8")
                if old_text in content:
                    push_undo(filepath, content)
                    new_content = content.replace(old_text, new_text)
                    p.write_text(new_content, encoding="utf-8")
                    changed.append(filepath)
        except Exception as e:
            errors.append(f"{filepath}: {e}")
    result = f"✅ Changed {len(changed)} files:\n" + "\n".join(changed)
    if errors:
        result += f"\n❌ Errors:\n" + "\n".join(errors)
    if not changed and not errors:
        result = f"ℹ️ No files matched pattern '{pattern}' or text not found"
    return result

def execute_tool(name, args, extra_dispatch=None):
    dispatch = {
        "read_file":       lambda: read_file(args["path"], args.get("lines_from"), args.get("lines_to")),
        "write_file":      lambda: write_file(args["path"], args["content"]),
        "edit_file":       lambda: edit_file(args["path"], args["old_text"], args["new_text"]),
        "run_command":     lambda: run_command(args["command"], args.get("workdir")),
        "list_files":      lambda: list_files(args.get("path", "."), args.get("recursive", False), args.get("pattern")),
        "search_in_files": lambda: search_in_files(args["pattern"], args.get("path", "."), args.get("file_pattern")),
        "ssh_run":         lambda: ssh_run(args["host"], args["command"]),
        "ssh_upload":      lambda: ssh_upload(args["host"], args["local_path"], args["remote_path"]),
        "ssh_download":    lambda: ssh_download(args["host"], args["remote_path"], args["local_path"]),
        "ssh_read_file":   lambda: ssh_read_file(args["host"], args["path"]),
        "ssh_write_file":  lambda: ssh_write_file(args["host"], args["path"], args["content"]),
        # New tools
        "git_command":     lambda: git_command(args["command"], args.get("workdir")),
        "fetch_url":       lambda: fetch_url(args["url"]),
        "docker_run":      lambda: docker_run(args["command"], args.get("host")),
        "db_query":        lambda: db_query(args["connection"], args["sql"], args.get("params")),
        "read_env":        lambda: read_env(args.get("path", ".env")),
        "write_env":       lambda: write_env(args["values"], args.get("path", ".env")),
        "edit_files_glob": lambda: edit_files_glob(args["pattern"], args["old_text"], args["new_text"]),
        # Semantic search (available if index exists)
        "semantic_search": lambda: _semantic_search_tool(args.get("query", ""), args.get("top_k", 5)),
    }
    if extra_dispatch:
        dispatch.update(extra_dispatch)
    fn = dispatch.get(name)
    return fn() if fn else f"Unknown tool: {name}"


def _semantic_search_tool(query: str, top_k: int = 5) -> str:
    """Wrapper for the semantic_search function usable as an agent tool."""
    try:
        results = semantic_search(query, ".", top_k)
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"## {r['file']}:{r['start_line']} (score={r['score']:.3f})\n{r['text'][:400]}")
        return "\n\n---\n\n".join(lines)
    except FileNotFoundError:
        return "No semantic index found. Run `kimi.py --index` first."
    except Exception as e:
        return f"❌ Semantic search error: {e}"

# ── Project context auto-loader ───────────────────────────────────────────────
PROJECT_INDICATORS = {
    "package.json":    "Node.js",
    "requirements.txt":"Python",
    "pyproject.toml":  "Python",
    "Cargo.toml":      "Rust",
    "go.mod":          "Go",
    "pom.xml":         "Java (Maven)",
    "composer.json":   "PHP",
    "build.gradle":    "Java (Gradle)",
}

def load_project_context(workdir="."):
    ctx_parts = []
    detected_types = []

    # Detect project type
    for fname, proj_type in PROJECT_INDICATORS.items():
        if (Path(workdir) / fname).exists():
            detected_types.append(proj_type)

    if detected_types:
        ctx_parts.append(f"### Detected Project Type\n{', '.join(detected_types)}")

    # Read key context files
    for fname in ["README.md", "README.txt", "package.json", "pyproject.toml", "Cargo.toml"]:
        p = Path(workdir) / fname
        if p.exists():
            content = p.read_text(encoding="utf-8", errors="ignore")[:2000]
            ctx_parts.append(f"### {fname}\n{content}")

    if ctx_parts:
        return "\n\n".join(ctx_parts)
    return None

# ── History / Sessions ────────────────────────────────────────────────────────
def get_session_file(session_name=None):
    if session_name:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        return SESSIONS_DIR / f"{session_name}.pkl"
    return HISTORY_FILE

def load_history(session_name=None):
    f = get_session_file(session_name)
    if f.exists():
        try:
            return pickle.loads(f.read_bytes())
        except:
            pass
    return []

def save_history(messages, session_name=None):
    f = get_session_file(session_name)
    try:
        f.write_bytes(pickle.dumps(messages[-50:]))  # keep last 50
    except:
        pass

def list_sessions():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = sorted(SESSIONS_DIR.glob("*.pkl"))
    if not sessions:
        return "No saved sessions."
    lines = []
    for s in sessions:
        try:
            msgs = pickle.loads(s.read_bytes())
            lines.append(f"  {s.stem:20s} ({len(msgs)} messages)")
        except:
            lines.append(f"  {s.stem:20s} (corrupted)")
    return "Saved sessions:\n" + "\n".join(lines)

# ── Token tracking ────────────────────────────────────────────────────────────
total_tokens = {"input": 0, "output": 0}
turn_tokens  = {"input": 0, "output": 0}

def track_tokens(usage):
    if usage:
        inp = getattr(usage, "prompt_tokens", 0)
        out = getattr(usage, "completion_tokens", 0)
        total_tokens["input"]  += inp
        total_tokens["output"] += out
        turn_tokens["input"]   += inp
        turn_tokens["output"]  += out

def cost_summary():
    inp  = total_tokens["input"]
    out  = total_tokens["output"]
    cost = (inp / 1000 * COST_PER_1K_INPUT) + (out / 1000 * COST_PER_1K_OUTPUT)
    return f"💰 Session: {inp}↑ {out}↓ tokens · ~${cost:.4f} (Kimi K2 via NIM)"

# ── Task planner ──────────────────────────────────────────────────────────────
PLAN_TRIGGER_WORDS = {"build", "create", "refactor", "implement", "develop", "redesign",
                      "migrate", "rewrite", "setup", "configure", "deploy", "integrate"}

def should_plan(text):
    words = set(text.lower().split())
    return len(text) > 100 or bool(words & PLAN_TRIGGER_WORDS)

def run_planner(user_input, model, system_content):
    """Make a planning call and return the plan text."""
    plan_system = (
        "You are a task planner. Given the user's request, output ONLY a concise numbered plan "
        "(3-7 steps) of what you will do. No code yet, just the plan. Be specific and actionable."
    )
    messages = [
        {"role": "system", "content": plan_system},
        {"role": "user", "content": user_input}
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            stream=False,
        )
        if resp.usage:
            track_tokens(resp.usage)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return None

# ── Image encoding ────────────────────────────────────────────────────────────
def encode_image(image_path_or_url):
    """Return (url_or_data_url, mime_type) for vision content."""
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        return image_path_or_url, None
    # Local file → base64
    p = Path(image_path_or_url)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path_or_url}")
    suffix = p.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".gif": "image/gif", ".webp": "image/webp"}
    mime = mime_map.get(suffix, "image/jpeg")
    data = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}", mime

# ── Markdown export ───────────────────────────────────────────────────────────
def save_session_markdown(messages):
    """Export conversation to a markdown file."""
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    filename = f"kimi-session-{now}.md"
    lines = [f"# Kimi Session — {now}\n"]
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "system":
            continue
        if isinstance(content, list):
            # Vision content
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(text_parts)
        if role == "assistant":
            lines.append(f"## 🤖 Assistant\n\n{content}\n")
        elif role == "user":
            lines.append(f"## 👤 You\n\n{content}\n")
        elif role == "tool":
            lines.append(f"### 🔧 Tool Result\n\n```\n{content[:500]}\n```\n")
    lines.append(f"\n---\n{cost_summary()}\n")
    Path(filename).write_text("\n".join(lines), encoding="utf-8")
    return filename

# ── Agentic loop (with streaming) ─────────────────────────────────────────────
def run_agent(messages, model, max_iterations=20, extra_tools=None, extra_dispatch=None,
              quiet=False):
    """
    Run the agentic loop.

    Args:
        messages:       Conversation history (mutated in place).
        model:          Model name to use.
        max_iterations: Maximum tool-call iterations before giving up.
        extra_tools:    Additional tool definitions to inject.
        extra_dispatch: Dict mapping tool name → callable for extra tools.
        quiet:          Suppress live streaming output (used by worker agents).
    """
    global interrupted
    if not quiet:
        interrupted = False
    turn_tokens["input"] = 0
    turn_tokens["output"] = 0

    tools_to_use = TOOLS + (extra_tools or [])

    for i in range(max_iterations):
        if interrupted:
            return "⚡ Interrupted by user."

        # Stream the response
        collected = ""
        tool_calls_raw = {}

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools_to_use,
                tool_choice="auto",
                max_tokens=8192,
                stream=True,
            )

            def _process_stream(stream):
                nonlocal collected
                if quiet:
                    for chunk in stream:
                        delta = chunk.choices[0].delta if chunk.choices else None
                        if not delta:
                            continue
                        if delta.content:
                            collected += delta.content
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in tool_calls_raw:
                                    tool_calls_raw[idx] = {"id": "", "name": "", "args": ""}
                                if tc.id:
                                    tool_calls_raw[idx]["id"] = tc.id
                                if tc.function:
                                    if tc.function.name:
                                        tool_calls_raw[idx]["name"] += tc.function.name
                                    if tc.function.arguments:
                                        tool_calls_raw[idx]["args"] += tc.function.arguments
                        if hasattr(chunk, "usage") and chunk.usage:
                            track_tokens(chunk.usage)
                else:
                    with Live(console=console, refresh_per_second=15) as live:
                        for chunk in stream:
                            if interrupted:
                                break
                            delta = chunk.choices[0].delta if chunk.choices else None
                            if not delta:
                                continue
                            if delta.content:
                                collected += delta.content
                                live.update(Text(collected))
                            if delta.tool_calls:
                                for tc in delta.tool_calls:
                                    idx = tc.index
                                    if idx not in tool_calls_raw:
                                        tool_calls_raw[idx] = {"id": "", "name": "", "args": ""}
                                    if tc.id:
                                        tool_calls_raw[idx]["id"] = tc.id
                                    if tc.function:
                                        if tc.function.name:
                                            tool_calls_raw[idx]["name"] += tc.function.name
                                        if tc.function.arguments:
                                            tool_calls_raw[idx]["args"] += tc.function.arguments
                            if hasattr(chunk, "usage") and chunk.usage:
                                track_tokens(chunk.usage)

            _process_stream(stream)

        except Exception as e:
            return f"❌ API Error: {e}"

        # No tool calls → done
        if not tool_calls_raw:
            messages.append({"role": "assistant", "content": collected})
            return collected

        # Build assistant message with tool calls
        tool_calls_list = []
        for idx in sorted(tool_calls_raw.keys()):
            tc = tool_calls_raw[idx]
            tool_calls_list.append({
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["args"]}
            })

        messages.append({
            "role": "assistant",
            "content": collected or None,
            "tool_calls": tool_calls_list
        })

        # Execute tools
        for tc in tool_calls_list:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except:
                args = {}

            # Mask secrets in env tool logs
            display_args = dict(args)
            if name == "write_env" and "values" in display_args:
                secret_kws = {"key", "secret", "password", "passwd", "token", "api", "auth", "credential"}
                display_args["values"] = {
                    k: ("***" if any(kw in k.lower() for kw in secret_kws) else v)
                    for k, v in display_args["values"].items()
                }

            if not quiet:
                console.print(f"\n[dim cyan]🔧 {name}({', '.join(f'{k}={repr(v)[:60]}' for k,v in display_args.items())})[/dim cyan]")

            result = execute_tool(name, args, extra_dispatch=extra_dispatch)
            short = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            if not quiet:
                console.print(f"[dim]   → {short}[/dim]")

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(result)
            })

    return "⚠️ Max iterations reached."

# ── System prompt ─────────────────────────────────────────────────────────────
def make_system_prompt(workdir, project_ctx):
    base = f"""You are Kimi, an expert coding assistant. Working directory: {Path(workdir).resolve()}

You have tools to: read/write/edit files, run shell commands, list directories, search in files,
run git commands, fetch URLs, run docker, query databases, manage .env files, and edit multiple files at once.

Workflow:
1. Explore first (list_files, read_file key files)
2. Make precise, targeted edits
3. Run tests/builds to verify
4. Report clearly what changed and why

Be concise. Show diffs when editing. Verify your changes work."""

    if project_ctx:
        base += f"\n\n## Project Context\n{project_ctx}"
    return base

# ── Orchestrator mode ─────────────────────────────────────────────────────────

MAX_SUBTASKS = 10

ORCHESTRATOR_PLANNER_SYSTEM = """You are an expert task orchestrator. Your job is to decompose a complex task into smaller, parallelizable subtasks.

Output ONLY valid JSON (no markdown fences, no explanation) in this exact format:
{
  "subtasks": [
    {
      "id": 1,
      "title": "Short title",
      "description": "Detailed description of what to do",
      "dependencies": []
    },
    {
      "id": 2,
      "title": "Another subtask",
      "description": "...",
      "dependencies": [1]
    }
  ]
}

Rules:
- Maximum """ + str(MAX_SUBTASKS) + """ subtasks
- dependencies is a list of subtask ids that must complete BEFORE this one starts
- Maximize parallelism: tasks with no shared dependencies can run simultaneously
- Each subtask should be self-contained and actionable
- Be specific: include filenames, function names, etc. when known
- If the task is simple (1-2 steps), use 1-2 subtasks"""

ORCHESTRATOR_AGGREGATOR_SYSTEM = """You are an expert at synthesizing results from parallel worker agents.

You will receive the original task and the results from multiple worker agents that each handled a subtask.
Your job is to:
1. Synthesize a clear, concise summary of what was accomplished
2. Highlight any conflicts, issues, or incomplete work
3. Suggest follow-up steps if needed

Be direct and actionable."""

# spawn_worker tool definition (for orchestrator to dynamically create subtasks)
SPAWN_WORKER_TOOL = {
    "type": "function",
    "function": {
        "name": "spawn_worker",
        "description": "Dynamically spawn an additional worker agent to handle a new subtask that wasn't in the original plan. Use this when you discover additional work that needs to be done.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short title for the subtask"},
                "description": {"type": "string", "description": "Detailed description of what the worker should do"},
                "context": {"type": "string", "description": "Any additional context or results from previous workers that this worker needs"}
            },
            "required": ["title", "description"]
        }
    }
}


def _topological_sort(subtasks: list) -> list:
    """Return subtasks in topological order (dependencies first)."""
    id_to_task = {t["id"]: t for t in subtasks}
    visited = set()
    result = []

    def visit(task_id):
        if task_id in visited:
            return
        visited.add(task_id)
        for dep in id_to_task.get(task_id, {}).get("dependencies", []):
            visit(dep)
        result.append(id_to_task[task_id])

    for task in subtasks:
        visit(task["id"])
    return result


def _render_orchestrator_status(task: str, subtasks: list, running: set, done: set, failed: set):
    """Render a progress view of the orchestration."""
    lines = [f"[bold cyan]🎯 Orchestrating:[/bold cyan] [bold]\"{task}\"[/bold]\n"]
    lines.append("[bold]Plan:[/bold]")
    for st in subtasks:
        sid = st["id"]
        title = st["title"]
        if sid in done:
            lines.append(f"  [green]✅ [{sid}] {title}[/green]")
        elif sid in failed:
            lines.append(f"  [red]❌ [{sid}] {title}[/red]")
        elif sid in running:
            lines.append(f"  [yellow]⚡ [{sid}] {title}[/yellow]")
        else:
            deps = st.get("dependencies", [])
            if deps:
                lines.append(f"  [dim]⏳ [{sid}] {title} (needs: {deps})[/dim]")
            else:
                lines.append(f"  [dim]⏳ [{sid}] {title}[/dim]")
    return "\n".join(lines)


def run_worker(subtask: dict, original_task: str, system_content: str,
               model: str, dep_results: dict) -> dict:
    """Run a single worker agent for a subtask. Returns result dict."""
    sid = subtask["id"]
    title = subtask["title"]
    desc = subtask["description"]

    # Build worker system + context
    worker_system = (
        f"{system_content}\n\n"
        f"## Orchestration Context\n"
        f"You are Worker #{sid} in a multi-agent orchestration system.\n"
        f"Overall task: {original_task}\n"
        f"Your specific subtask: {title}\n"
    )

    if dep_results:
        worker_system += "\n## Results from prerequisite subtasks:\n"
        for dep_id, dep_result in dep_results.items():
            worker_system += f"\n### Subtask {dep_id} result:\n{dep_result}\n"

    worker_messages = [
        {"role": "system", "content": worker_system},
        {"role": "user", "content": f"Please complete this subtask:\n\n{desc}"}
    ]

    try:
        result = run_agent(worker_messages, model, max_iterations=15, quiet=True)
        return {"id": sid, "title": title, "status": "done", "result": result}
    except Exception as e:
        return {"id": sid, "title": title, "status": "failed", "result": str(e)}


def run_orchestrator(task: str, system_content: str, orchestrator_model: str,
                     worker_model: str, max_workers: int = 4):
    """
    Main orchestration loop:
    1. Plan: decompose task into subtasks
    2. Execute: run subtasks in parallel respecting dependencies
    3. Aggregate: synthesize results
    """
    console.print(Panel(
        f"[bold cyan]🎯 Kimi Orchestrator[/bold cyan]\n"
        f"[dim]Planner: {orchestrator_model}\nWorker: {worker_model}\nMax workers: {max_workers}[/dim]",
        border_style="cyan"
    ))
    console.print(f"\n[bold]Task:[/bold] {task}\n")

    # ── Step 1: Plan ──────────────────────────────────────────────────────────
    console.print("[dim cyan]📋 Planning task decomposition...[/dim cyan]")
    plan_messages = [
        {"role": "system", "content": ORCHESTRATOR_PLANNER_SYSTEM},
        {"role": "user", "content": f"Task: {task}\n\nWorking directory context:\n{system_content[:1000]}"}
    ]

    subtasks = []
    try:
        # Try orchestrator model first, fall back to fast
        for attempt_model in [orchestrator_model, MODELS["fast"]]:
            try:
                resp = client.chat.completions.create(
                    model=attempt_model,
                    messages=plan_messages,
                    max_tokens=2048,
                    stream=False,
                )
                if resp.usage:
                    track_tokens(resp.usage)
                raw = resp.choices[0].message.content.strip()

                # Strip markdown fences if present
                raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
                raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)

                plan_data = json.loads(raw)
                subtasks = plan_data.get("subtasks", [])
                if subtasks:
                    break
            except (json.JSONDecodeError, Exception) as e:
                console.print(f"[yellow]⚠️ Plan attempt failed with {attempt_model}: {e}[/yellow]")
                continue

        if not subtasks:
            console.print("[red]❌ Could not generate a plan. Running as single task.[/red]")
            # Fallback: single subtask
            subtasks = [{"id": 1, "title": task[:60], "description": task, "dependencies": []}]

        # Clamp to max subtasks
        if len(subtasks) > MAX_SUBTASKS:
            console.print(f"[yellow]⚠️ Plan has {len(subtasks)} subtasks (max {MAX_SUBTASKS}), truncating.[/yellow]")
            subtasks = subtasks[:MAX_SUBTASKS]

    except Exception as e:
        console.print(f"[red]❌ Planner error: {e}[/red]")
        subtasks = [{"id": 1, "title": task[:60], "description": task, "dependencies": []}]

    # ── Display plan ──────────────────────────────────────────────────────────
    plan_lines = [f"[bold]📋 Plan ({len(subtasks)} subtasks):[/bold]"]
    for st in subtasks:
        deps = st.get("dependencies", [])
        dep_str = f" [dim](after {deps})[/dim]" if deps else ""
        plan_lines.append(f"  [{st['id']}] {st['title']}{dep_str}")
    console.print("\n".join(plan_lines))
    console.print()

    # ── Step 2: Execute with dependency resolution ────────────────────────────
    sorted_tasks = _topological_sort(subtasks)
    id_to_task = {t["id"]: t for t in subtasks}

    running: set = set()
    done: set = set()
    failed: set = set()
    results: dict = {}  # id → result string
    pending_futures: dict = {}  # future → subtask id

    # For dynamic spawn_worker calls from orchestrator
    dynamic_tasks = []
    dynamic_lock = threading.Lock()
    next_dynamic_id = [max(t["id"] for t in subtasks) + 1]

    def spawn_worker_fn(title: str, description: str, context: str = "") -> str:
        """Called when orchestrator wants to dynamically spawn a new worker."""
        with dynamic_lock:
            new_id = next_dynamic_id[0]
            next_dynamic_id[0] += 1
        new_task = {
            "id": new_id,
            "title": title,
            "description": description + (f"\n\nContext:\n{context}" if context else ""),
            "dependencies": list(done),  # depends on all completed tasks
        }
        with dynamic_lock:
            dynamic_tasks.append(new_task)
            sorted_tasks.append(new_task)
            id_to_task[new_id] = new_task
            subtasks.append(new_task)
        console.print(f"[cyan]  ➕ Dynamically spawned worker [{new_id}]: {title}[/cyan]")
        return f"✅ Spawned worker [{new_id}]: {title}"

    def can_run(subtask_item):
        return all(dep in done for dep in subtask_item.get("dependencies", []))

    import time as _time

    with Live(console=console, refresh_per_second=4) as live:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            processed = set()

            while True:
                # Find tasks ready to run
                for subtask_item in list(sorted_tasks):
                    tid = subtask_item["id"]
                    if tid in done or tid in failed or tid in running or tid in processed:
                        continue
                    if can_run(subtask_item):
                        running.add(tid)
                        processed.add(tid)
                        dep_results = {
                            dep: results[dep]
                            for dep in subtask_item.get("dependencies", [])
                            if dep in results
                        }
                        future = executor.submit(
                            run_worker, subtask_item, task, system_content, worker_model, dep_results
                        )
                        pending_futures[future] = tid

                # Update display
                live.update(Text.from_markup(
                    _render_orchestrator_status(task, subtasks, running, done, failed)
                ))

                if not pending_futures:
                    if all(t["id"] in done or t["id"] in failed for t in sorted_tasks):
                        break
                    # Check for dynamic tasks that appeared
                    with dynamic_lock:
                        new_dynamic = [t for t in dynamic_tasks if t["id"] not in processed and can_run(t)]
                    if not new_dynamic:
                        break
                    _time.sleep(0.25)
                    continue

                # Wait for any future to complete
                done_futures, _ = concurrent.futures.wait(
                    list(pending_futures.keys()),
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )

                for future in done_futures:
                    tid = pending_futures.pop(future)
                    running.discard(tid)
                    try:
                        worker_result = future.result()
                        results[tid] = worker_result.get("result", "")
                        if worker_result.get("status") == "done":
                            done.add(tid)
                        else:
                            failed.add(tid)
                    except Exception as e:
                        failed.add(tid)
                        results[tid] = f"❌ Worker crashed: {e}"

                # Refresh display with updated task list
                live.update(Text.from_markup(
                    _render_orchestrator_status(task, subtasks, running, done, failed)
                ))

    # Final status line
    console.print()
    status_parts = []
    if done:
        status_parts.append(f"[green]✅ {len(done)} completed[/green]")
    if failed:
        status_parts.append(f"[red]❌ {len(failed)} failed[/red]")
    console.print("  ".join(status_parts))
    console.print()

    # ── Step 3: Aggregate ─────────────────────────────────────────────────────
    console.print("[dim cyan]🧠 Synthesizing results...[/dim cyan]\n")

    agg_content = f"Original task: {task}\n\nWorker results:\n\n"
    for st in subtasks:
        sid = st["id"]
        status_emoji = "✅" if sid in done else "❌"
        agg_content += f"## {status_emoji} Subtask {sid}: {st['title']}\n"
        agg_content += f"{results.get(sid, 'No result')}\n\n"

    agg_messages = [
        {"role": "system", "content": ORCHESTRATOR_AGGREGATOR_SYSTEM},
        {"role": "user", "content": agg_content}
    ]

    try:
        agg_resp = client.chat.completions.create(
            model=orchestrator_model,
            messages=agg_messages,
            max_tokens=4096,
            stream=True,
        )
        final_summary = ""
        with Live(console=console, refresh_per_second=15) as live:
            for chunk in agg_resp:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    final_summary += delta.content
                    live.update(Text(final_summary))
                if hasattr(chunk, "usage") and chunk.usage:
                    track_tokens(chunk.usage)
    except Exception as e:
        final_summary = f"❌ Aggregation failed: {e}\n\nIndividual results:\n{agg_content}"
        console.print(final_summary)

    console.print()
    return final_summary


# ── GitHub Issues Integration ─────────────────────────────────────────────────

def _gh(cmd: str) -> str:
    """Run a gh CLI command and return stdout."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"gh exit {result.returncode}")
    return result.stdout.strip()


def _parse_repo_from_remote(remote_url: str) -> tuple:
    """Parse owner/repo from a git remote URL (HTTPS or SSH)."""
    # SSH: git@github.com:owner/repo.git
    m = re.match(r'git@[^:]+:([^/]+)/([^/]+?)(?:\.git)?$', remote_url)
    if m:
        return m.group(1), m.group(2)
    # HTTPS: https://github.com/owner/repo.git
    m = re.match(r'https?://[^/]+/([^/]+)/([^/]+?)(?:\.git)?$', remote_url)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(f"Cannot parse repo from remote URL: {remote_url}")


def run_github_issue(issue_num: int, model: str, messages: list,
                     auto_pr: bool, args) -> None:
    """Fetch a GitHub issue, run agent to fix it, optionally create PR."""
    console.print(f"[cyan]🐙 Loading GitHub issue #{issue_num}...[/cyan]")

    # Detect repo
    try:
        remote_url = _gh("git remote get-url origin")
        owner, repo = _parse_repo_from_remote(remote_url)
        console.print(f"[dim]   Repo: {owner}/{repo}[/dim]")
    except Exception as e:
        console.print(f"[red]❌ Could not detect repo: {e}[/red]")
        return

    # Fetch issue
    try:
        issue_json = _gh(f"gh api repos/{owner}/{repo}/issues/{issue_num}")
        issue = json.loads(issue_json)
    except Exception as e:
        console.print(f"[red]❌ Could not fetch issue: {e}[/red]")
        return

    # Fetch comments
    comments_text = ""
    try:
        comments_json = _gh(f"gh api repos/{owner}/{repo}/issues/{issue_num}/comments")
        comments = json.loads(comments_json)
        if comments:
            parts = []
            for c in comments:
                author = c.get("user", {}).get("login", "unknown")
                body = c.get("body", "").strip()
                parts.append(f"**{author}:** {body}")
            comments_text = "\n\n".join(parts)
        else:
            comments_text = "(no comments)"
    except Exception as e:
        comments_text = f"(could not fetch comments: {e})"

    issue_title = issue.get("title", f"Issue #{issue_num}")
    issue_body  = issue.get("body") or "(no description)"

    prompt = (
        f"Fix GitHub issue #{issue_num}: {issue_title}\n\n"
        f"Description:\n{issue_body}\n\n"
        f"Comments:\n{comments_text}"
    )

    console.print(Panel(
        f"[bold]#{issue_num}:[/bold] {issue_title}",
        title="[bold cyan]🐙 GitHub Issue[/bold cyan]",
        border_style="cyan"
    ))

    # Task planning
    if not args.no_plan and should_plan(prompt):
        console.print("[dim cyan]📋 Planning task...[/dim cyan]")
        system_content = messages[0]["content"] if messages else ""
        plan = run_planner(prompt, model, system_content)
        if plan:
            console.print(Panel(plan, title="[bold]📋 Plan[/bold]", border_style="blue"))
            console.print()

    messages.append({"role": "user", "content": prompt})
    result = run_agent(messages, model)
    console.print()

    # PR creation
    if auto_pr:
        create_pr = True
    else:
        try:
            answer = input("Create PR? (y/n): ").strip().lower()
            create_pr = answer in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            create_pr = False

    if create_pr:
        console.print("[cyan]🔀 Creating PR...[/cyan]")
        try:
            pr_title = f"fix: {issue_title}"
            pr_body  = f"Closes #{issue_num}"
            pr_out = _gh(
                f'gh pr create --title {json.dumps(pr_title)} --body {json.dumps(pr_body)}'
            )
            console.print(f"[green]✅ PR created: {pr_out}[/green]")
        except Exception as e:
            console.print(f"[red]❌ PR creation failed: {e}[/red]")


# ── Semantic Search ───────────────────────────────────────────────────────────

EMBED_MODEL  = "nvidia/llama-3.2-nv-embedqa-1b-v2"
CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
                   ".java", ".php", ".rb", ".cpp", ".c", ".h"}
INDEX_FILE = ".kimi_index.pkl"
SKIP_DIRS  = {"node_modules", ".git", "dist", "__pycache__", ".kimi_index.pkl"}
CHUNK_SIZE = 60   # lines per chunk
CHUNK_OVERLAP = 10


def _embed(texts: list) -> list:
    """Call NVIDIA embeddings API and return list of float vectors."""
    payload = json.dumps({
        "model": EMBED_MODEL,
        "input": texts,
        "encoding_format": "float"
    }).encode("utf-8")
    req = urllib_request.Request(
        "https://integrate.api.nvidia.com/v1/embeddings",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [item["embedding"] for item in data["data"]]


def _cosine_sim(v1: list, v2: list) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = sum(x * x for x in v1) ** 0.5
    mag2 = sum(x * x for x in v2) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def build_index(workdir: str = ".") -> None:
    """Walk code files, chunk them, embed, and save to INDEX_FILE."""
    root = Path(workdir).resolve()
    chunks = []

    # Collect all code files
    all_files = []
    for p in root.rglob("*"):
        # Skip unwanted dirs
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        if p.is_file() and p.suffix in CODE_EXTENSIONS:
            all_files.append(p)

    console.print(f"[cyan]📁 Found {len(all_files)} code files. Embedding...[/cyan]")

    for fpath in all_files:
        try:
            lines = fpath.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        rel = str(fpath.relative_to(root))
        # Chunk with overlap
        start = 0
        while start < len(lines):
            end = min(start + CHUNK_SIZE, len(lines))
            chunk_text = "\n".join(lines[start:end])
            if chunk_text.strip():
                chunks.append({"file": rel, "start_line": start + 1, "text": chunk_text, "embedding": None})
            start += CHUNK_SIZE - CHUNK_OVERLAP

    console.print(f"[cyan]🔢 {len(chunks)} chunks to embed...[/cyan]")

    # Batch-embed (NVIDIA API supports up to ~16 inputs at a time)
    BATCH = 8
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        texts = [c["text"] for c in batch]
        try:
            embeddings = _embed(texts)
            for c, emb in zip(batch, embeddings):
                c["embedding"] = emb
            console.print(f"[dim]  Embedded {min(i + BATCH, len(chunks))}/{len(chunks)}[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Embedding batch {i//BATCH} failed: {e}[/yellow]")
            for c in batch:
                c["embedding"] = []

    # Save
    index_path = root / INDEX_FILE
    index_path.write_bytes(pickle.dumps(chunks))
    console.print(f"[green]✅ Index saved: {len(chunks)} chunks → {index_path}[/green]")


def semantic_search(query: str, workdir: str = ".", top_k: int = 5) -> list:
    """Search the semantic index for query. Returns top_k chunks."""
    root = Path(workdir).resolve()
    index_path = root / INDEX_FILE

    if not index_path.exists():
        raise FileNotFoundError(f"No index found at {index_path}. Run with --index first.")

    chunks = pickle.loads(index_path.read_bytes())
    # Filter out chunks with no embeddings
    chunks = [c for c in chunks if c.get("embedding")]

    if not chunks:
        return []

    try:
        query_embedding = _embed([query])[0]
    except Exception as e:
        raise RuntimeError(f"Could not embed query: {e}")

    scored = [
        (c, _cosine_sim(query_embedding, c["embedding"]))
        for c in chunks
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"file": c["file"], "start_line": c["start_line"], "text": c["text"], "score": s}
            for c, s in scored[:top_k]]


# Tool definition for semantic_search (added dynamically when index exists)
SEMANTIC_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "semantic_search",
        "description": "Semantic search over indexed code files. Returns relevant code chunks by meaning.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query"},
                "top_k": {"type": "integer", "description": "Number of results (default 5)", "default": 5}
            },
            "required": ["query"]
        }
    }
}


# ── Rich TUI ──────────────────────────────────────────────────────────────────

def run_tui(model: str, messages: list, session_name, args) -> None:
    """Enhanced interactive mode using rich Layout + Live."""

    recent_tools: list = []   # list of (tool_name, short_result)
    conversation: list = []   # list of (role, text)

    def _make_layout() -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=1),
        )
        layout["body"].split_row(
            Layout(name="tools", ratio=1),
            Layout(name="chat",  ratio=2),
        )
        return layout

    def _header_panel() -> Panel:
        inp  = total_tokens["input"]
        out  = total_tokens["output"]
        sess = session_name or "default"
        return Panel(
            f"[bold cyan]Kimi CLI[/bold cyan]  "
            f"[dim]model:[/dim] [yellow]{model}[/yellow]  "
            f"[dim]session:[/dim] [green]{sess}[/green]  "
            f"[dim]tokens:[/dim] {inp}↑ {out}↓",
            border_style="cyan",
        )

    def _tools_panel() -> Panel:
        lines = []
        for tname, tres in recent_tools[-12:]:
            lines.append(f"[cyan]{tname}[/cyan]\n[dim]{tres[:60]}[/dim]")
        body = "\n".join(lines) if lines else "[dim]No tool calls yet[/dim]"
        return Panel(body, title="[bold]🔧 Tool Calls[/bold]", border_style="blue")

    def _chat_panel() -> Panel:
        lines = []
        for role, text in conversation[-20:]:
            if role == "user":
                lines.append(f"[bold green]You:[/bold green] {text[:200]}")
            else:
                lines.append(f"[bold cyan]Kimi:[/bold cyan] {text[:300]}")
        body = "\n\n".join(lines) if lines else "[dim]Start typing...[/dim]"
        return Panel(body, title="[bold]💬 Conversation[/bold]", border_style="green")

    def _footer() -> str:
        return "[dim]Type your task · 'exit' quit · 'clear' reset · 'undo' revert · 'save' export[/dim]"

    layout = _make_layout()

    def _refresh_layout():
        layout["header"].update(_header_panel())
        layout["body"]["tools"].update(_tools_panel())
        layout["body"]["chat"].update(_chat_panel())
        layout["footer"].update(_footer())

    # Patch run_agent to capture tool calls for display
    original_execute_tool = execute_tool

    def _patched_execute_tool(name, tool_args, extra_dispatch=None):
        result = original_execute_tool(name, tool_args, extra_dispatch=extra_dispatch)
        short = str(result)[:80].replace("\n", " ")
        recent_tools.append((name, short))
        return result

    console.print("[cyan]🖥️  Kimi TUI — press Ctrl+C or type 'exit' to quit[/cyan]\n")

    with Live(layout, console=console, refresh_per_second=4, screen=False) as live:
        _refresh_layout()
        live.update(layout)

        while True:
            live.stop()
            try:
                user_input = input("\n[You] > ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            live.start()

            if user_input.lower() in ("exit", "quit", "bye"):
                break
            if user_input.lower() == "clear":
                messages[:] = [messages[0]]
                conversation.clear()
                recent_tools.clear()
                f = get_session_file(session_name)
                f.unlink(missing_ok=True)
                _refresh_layout()
                continue
            if user_input.lower() == "undo":
                if undo_stack:
                    path, original = undo_stack.pop()
                    try:
                        Path(path).write_text(original, encoding="utf-8")
                        conversation.append(("assistant", f"✅ Restored {path}"))
                    except Exception as e:
                        conversation.append(("assistant", f"❌ Undo failed: {e}"))
                else:
                    conversation.append(("assistant", "⚠️ Nothing to undo."))
                _refresh_layout()
                continue
            if user_input.lower() == "save":
                try:
                    fname = save_session_markdown(messages)
                    conversation.append(("assistant", f"✅ Saved to {fname}"))
                except Exception as e:
                    conversation.append(("assistant", f"❌ Save failed: {e}"))
                _refresh_layout()
                continue
            if not user_input:
                continue

            conversation.append(("user", user_input))
            messages.append({"role": "user", "content": user_input})
            _refresh_layout()
            live.update(layout)

            # Run agent
            try:
                result = run_agent(messages, model)
            except Exception as e:
                result = f"❌ Error: {e}"

            conversation.append(("assistant", result or "(done)"))

            t_in  = turn_tokens["input"]
            t_out = turn_tokens["output"]
            turn_cost = (t_in / 1000 * COST_PER_1K_INPUT) + (t_out / 1000 * COST_PER_1K_OUTPUT)
            console.print(f"\n[dim]💬 Turn: {t_in}↑ {t_out}↓ · ~${turn_cost:.4f}[/dim]")

            save_history(messages, session_name)
            _refresh_layout()
            live.update(layout)

    console.print(f"\n[dim]{cost_summary()} · Bye! 👋[/dim]")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Kimi Coding CLI")
    parser.add_argument("prompt", nargs="?", help="Task (omit for interactive mode)")
    parser.add_argument("--workdir", "-w", default=".", help="Working directory")
    parser.add_argument("--model", "-m", default="fast",
                        help=f"Model alias or full name. Aliases: {list(MODELS.keys())}")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume last session")
    parser.add_argument("--clear", action="store_true", help="Clear history and exit")
    parser.add_argument("--session", "-s", default=None, help="Named session (stored in ~/.kimi_sessions/)")
    parser.add_argument("--image", "-i", default=None, help="Image path or URL to send with prompt (vision)")
    parser.add_argument("--no-plan", action="store_true", help="Skip task planning step")
    parser.add_argument("--orchestrate", "-O", metavar="TASK",
                        help="Activate multi-agent orchestration mode for a complex task")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for orchestration (default: 4, max: 8)")
    parser.add_argument("--issue", "-I", type=int, default=None, metavar="N",
                        help="Load GitHub issue #N as prompt and optionally create a PR after completion")
    parser.add_argument("--pr", action="store_true",
                        help="Auto-create a PR after completing a GitHub issue (use with --issue)")
    parser.add_argument("--index", action="store_true",
                        help="Build semantic search index for code files in workdir")
    parser.add_argument("--search", default=None, metavar="QUERY",
                        help="Semantic search over indexed code files")
    parser.add_argument("--tui", action="store_true",
                        help="Launch enhanced rich-based interactive TUI mode")
    args = parser.parse_args()

    # Model resolution
    model = MODELS.get(args.model, args.model)

    # History
    if args.clear:
        f = get_session_file(args.session)
        f.unlink(missing_ok=True)
        console.print("[green]✅ History cleared.[/green]")
        return

    os.chdir(args.workdir)
    project_ctx = load_project_context(args.workdir)
    system_msg  = {"role": "system", "content": make_system_prompt(args.workdir, project_ctx)}

    messages = []
    if args.resume or args.session:
        messages = load_history(args.session)
        if messages:
            console.print(f"[dim]📂 Resumed {len(messages)} messages from session '{args.session or 'last'}'[/dim]")

    if not messages:
        messages = [system_msg]
    else:
        messages[0] = system_msg  # refresh system prompt

    # Header
    session_label = f" · session: {args.session}" if args.session else ""
    console.print(Panel(
        f"[bold cyan]Kimi Coding CLI[/bold cyan]\n"
        f"[dim]Model: {model}\n"
        f"Dir:   {Path(args.workdir).resolve()}"
        + (f"\nCtx:   {', '.join(['README','package.json','pyproject.toml'] if project_ctx else [])}" if project_ctx else "")
        + session_label
        + "[/dim]",
        border_style="cyan"
    ))

    def run_turn(user_input, image_path=None):
        # Build message content
        if image_path:
            try:
                img_url, mime = encode_image(image_path)
                content = [
                    {"type": "text", "text": user_input},
                    {"type": "image_url", "image_url": {"url": img_url}}
                ]
                messages.append({"role": "user", "content": content})
                console.print(f"[dim]🖼️  Image attached: {image_path}[/dim]")
            except Exception as e:
                console.print(f"[yellow]⚠️ Could not attach image: {e}. Sending text only.[/yellow]")
                messages.append({"role": "user", "content": user_input})
        else:
            messages.append({"role": "user", "content": user_input})

        console.print()

        # Task planning
        if not args.no_plan and should_plan(user_input):
            console.print("[dim cyan]📋 Planning task...[/dim cyan]")
            plan = run_planner(user_input, model, system_msg["content"])
            if plan:
                console.print(Panel(plan, title="[bold]📋 Plan[/bold]", border_style="blue"))
                console.print()

        result = run_agent(messages, model)
        console.print()

        # Cost per turn
        t_in  = turn_tokens["input"]
        t_out = turn_tokens["output"]
        turn_cost = (t_in / 1000 * COST_PER_1K_INPUT) + (t_out / 1000 * COST_PER_1K_OUTPUT)
        console.print(
            f"[dim]💬 Turn: {t_in}↑ {t_out}↓ · ~${turn_cost:.4f}[/dim]"
        )
        save_history(messages, args.session)
        return result

    # ── Semantic index build ──────────────────────────────────────────────────
    if args.index:
        build_index(args.workdir)
        console.print(f"\n[dim]{cost_summary()} · Index complete ✅[/dim]")
        return

    # ── Semantic search query ─────────────────────────────────────────────────
    if args.search:
        try:
            results = semantic_search(args.search, args.workdir)
            if not results:
                console.print("[yellow]No results found.[/yellow]")
            else:
                for i, r in enumerate(results, 1):
                    console.print(Panel(
                        r["text"][:600],
                        title=f"[bold]#{i} {r['file']}:{r['start_line']} (score={r['score']:.3f})[/bold]",
                        border_style="blue"
                    ))
        except FileNotFoundError as e:
            console.print(f"[red]❌ {e}[/red]")
        except Exception as e:
            console.print(f"[red]❌ Search error: {e}[/red]")
        return

    # ── GitHub Issue mode ─────────────────────────────────────────────────────
    if args.issue is not None:
        run_github_issue(args.issue, model, messages, args.pr, args)
        save_history(messages, args.session)
        console.print(f"\n[dim]{cost_summary()} · Issue #{args.issue} complete 🐙[/dim]")
        return

    # ── TUI mode ──────────────────────────────────────────────────────────────
    if args.tui:
        run_tui(model, messages, args.session, args)
        return

    # ── Orchestrator mode ─────────────────────────────────────────────────────
    if args.orchestrate:
        orchestrate_task = args.orchestrate
        orchestrator_model = MODELS["orchestrator"]
        worker_model = MODELS["fast"]
        max_workers = max(1, min(8, args.workers))
        system_content = make_system_prompt(args.workdir, project_ctx)
        run_orchestrator(
            task=orchestrate_task,
            system_content=system_content,
            orchestrator_model=orchestrator_model,
            worker_model=worker_model,
            max_workers=max_workers,
        )
        console.print(f"\n[dim]{cost_summary()} · Orchestration complete 🎯[/dim]")
        return

    if args.prompt:
        console.print(f"[bold]> {args.prompt}[/bold]\n")
        run_turn(args.prompt, args.image)
    else:
        console.print(
            "[dim]Interactive — type your task, 'exit' to quit, 'clear' to reset, "
            "'undo' to revert last file, 'save' to export markdown, 'sessions' to list sessions[/dim]\n"
        )
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            except (KeyboardInterrupt, EOFError):
                break

            if user_input.lower() in ("exit", "quit", "bye"):
                break
            if user_input.lower() == "clear":
                messages[:] = [system_msg]
                f = get_session_file(args.session)
                f.unlink(missing_ok=True)
                console.print("[green]✅ Context cleared.[/green]")
                continue
            if user_input.lower() == "undo":
                if undo_stack:
                    path, original = undo_stack.pop()
                    try:
                        Path(path).write_text(original, encoding="utf-8")
                        console.print(f"[green]✅ Restored {path}[/green]")
                    except Exception as e:
                        console.print(f"[red]❌ Undo failed: {e}[/red]")
                else:
                    console.print("[yellow]⚠️ Nothing to undo.[/yellow]")
                continue
            if user_input.lower() == "save":
                try:
                    fname = save_session_markdown(messages)
                    console.print(f"[green]✅ Saved to {fname}[/green]")
                except Exception as e:
                    console.print(f"[red]❌ Save failed: {e}[/red]")
                continue
            if user_input.lower() == "sessions":
                console.print(list_sessions())
                continue
            if not user_input.strip():
                continue

            run_turn(user_input)

    # Session summary
    console.print(f"\n[dim]{cost_summary()} · Bye! 👋[/dim]")

if __name__ == "__main__":
    main()
