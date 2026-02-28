#!/usr/bin/env python3
"""
Kimi CLI - Agentic coding assistant powered by Kimi via NVIDIA NIM
Usage:
  python3 kimi.py "task"                  # one-shot
  python3 kimi.py -w /path/to/project     # interactive in project dir
  python3 kimi.py --model kimi-k2-thinking "hard task"
  python3 kimi.py --session myproject     # named session
  python3 kimi.py --image screenshot.png "what's in this image?"
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
    "fast":    "moonshotai/kimi-k2-instruct",
    "smart":   "moonshotai/kimi-k2.5",
    "think":   "moonshotai/kimi-k2-thinking",
    "latest":  "moonshotai/kimi-k2-instruct-0905",
}

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
        # Save to undo stack if file exists
        if p.exists():
            push_undo(p, p.read_text(encoding="utf-8"))
        p.write_text(content, encoding="utf-8")
        return f"✅ Written {len(content)} chars to {path}"
    except Exception as e:
        return f"❌ Error: {e}"

def edit_file(path, old_text, new_text):
    try:
        content = Path(path).read_text(encoding="utf-8")
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
        Path(path).write_text(new_content, encoding="utf-8")
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

def execute_tool(name, args):
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
    }
    fn = dispatch.get(name)
    return fn() if fn else f"Unknown tool: {name}"

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
def run_agent(messages, model, max_iterations=20):
    global interrupted
    interrupted = False
    turn_tokens["input"] = 0
    turn_tokens["output"] = 0

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
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=8192,
                stream=True,
            )

            with Live(console=console, refresh_per_second=15) as live:
                for chunk in stream:
                    if interrupted:
                        break
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    # Text streaming
                    if delta.content:
                        collected += delta.content
                        live.update(Text(collected))

                    # Tool call streaming
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

                    # Track usage
                    if hasattr(chunk, "usage") and chunk.usage:
                        track_tokens(chunk.usage)

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

            console.print(f"\n[dim cyan]🔧 {name}({', '.join(f'{k}={repr(v)[:60]}' for k,v in display_args.items())})[/dim cyan]")

            result = execute_tool(name, args)
            short = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
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
