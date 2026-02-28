#!/usr/bin/env python3
"""
Kimi CLI - Agentic coding assistant powered by Kimi via NVIDIA NIM
Usage:
  python3 kimi.py "task"                  # one-shot
  python3 kimi.py -w /path/to/project     # interactive in project dir
  python3 kimi.py --model kimi-k2-thinking "hard task"
"""

import os
import sys
import json
import subprocess
import argparse
import signal
import pickle
import tempfile
from pathlib import Path
from datetime import datetime
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

client  = OpenAI(api_key=API_KEY, base_url=BASE_URL)
console = Console()

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
            "description": "Write content to a file (creates or overwrites)",
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
            "description": "Replace specific text in a file (shows diff before applying)",
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
        p.write_text(content, encoding="utf-8")
        return f"✅ Written {len(content)} chars to {path}"
    except Exception as e:
        return f"❌ Error: {e}"

def edit_file(path, old_text, new_text):
    try:
        content = Path(path).read_text(encoding="utf-8")
        if old_text not in content:
            return f"❌ Text not found in {path}"
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
    target = _resolve_host(host)
    # Write locally to temp file, then SCP
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', delete=False, encoding='utf-8') as f:
        f.write(content)
        tmp = f.name
    try:
        result = ssh_upload(host, tmp, path)
        return result
    finally:
        os.unlink(tmp)

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
    }
    fn = dispatch.get(name)
    return fn() if fn else f"Unknown tool: {name}"

# ── Project context auto-loader ───────────────────────────────────────────────
def load_project_context(workdir="."):
    ctx_parts = []
    for fname in ["README.md", "README.txt", "package.json", "pyproject.toml", "Cargo.toml"]:
        p = Path(workdir) / fname
        if p.exists():
            content = p.read_text(encoding="utf-8", errors="ignore")[:2000]
            ctx_parts.append(f"### {fname}\n{content}")
    if ctx_parts:
        return "\n\n".join(ctx_parts)
    return None

# ── History ───────────────────────────────────────────────────────────────────
def load_history():
    if HISTORY_FILE.exists():
        try:
            return pickle.loads(HISTORY_FILE.read_bytes())
        except:
            pass
    return []

def save_history(messages):
    try:
        HISTORY_FILE.write_bytes(pickle.dumps(messages[-50:]))  # keep last 50
    except:
        pass

# ── Token tracking ────────────────────────────────────────────────────────────
total_tokens = {"input": 0, "output": 0}

def track_tokens(usage):
    if usage:
        total_tokens["input"]  += getattr(usage, "prompt_tokens", 0)
        total_tokens["output"] += getattr(usage, "completion_tokens", 0)

# ── Agentic loop (with streaming) ─────────────────────────────────────────────
def run_agent(messages, model, max_iterations=20):
    global interrupted
    interrupted = False

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

            console.print(f"\n[dim cyan]🔧 {name}({', '.join(f'{k}={repr(v)[:60]}' for k,v in args.items())})[/dim cyan]")

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

You have tools to: read/write/edit files, run shell commands, list directories, search in files.

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
    args = parser.parse_args()

    # Model resolution
    model = MODELS.get(args.model, args.model)

    # History
    if args.clear:
        HISTORY_FILE.unlink(missing_ok=True)
        console.print("[green]✅ History cleared.[/green]")
        return

    os.chdir(args.workdir)
    project_ctx = load_project_context(args.workdir)
    system_msg  = {"role": "system", "content": make_system_prompt(args.workdir, project_ctx)}

    messages = []
    if args.resume:
        messages = load_history()
        console.print(f"[dim]📂 Resumed {len(messages)} messages from last session[/dim]")

    if not messages:
        messages = [system_msg]
    else:
        messages[0] = system_msg  # refresh system prompt

    # Header
    console.print(Panel(
        f"[bold cyan]Kimi Coding CLI[/bold cyan]\n"
        f"[dim]Model: {model}\n"
        f"Dir:   {Path(args.workdir).resolve()}"
        + (f"\nCtx:   {', '.join(['README','package.json','pyproject.toml'] if project_ctx else [])}" if project_ctx else "")
        + "[/dim]",
        border_style="cyan"
    ))

    def run_turn(user_input):
        messages.append({"role": "user", "content": user_input})
        console.print()
        result = run_agent(messages, model)
        console.print()
        # Cost summary
        console.print(
            f"[dim]💬 {total_tokens['input']}↑ {total_tokens['output']}↓ tokens[/dim]"
        )
        save_history(messages)
        return result

    if args.prompt:
        console.print(f"[bold]> {args.prompt}[/bold]\n")
        run_turn(args.prompt)
    else:
        console.print("[dim]Interactive — type your task, 'exit' to quit, 'clear' to reset history[/dim]\n")
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            except (KeyboardInterrupt, EOFError):
                break

            if user_input.lower() in ("exit", "quit", "bye"):
                break
            if user_input.lower() == "clear":
                messages[:] = [system_msg]
                HISTORY_FILE.unlink(missing_ok=True)
                console.print("[green]✅ Context cleared.[/green]")
                continue
            if not user_input.strip():
                continue

            run_turn(user_input)

    console.print(f"\n[dim]Session total: {total_tokens['input']}↑ {total_tokens['output']}↓ tokens · Bye! 👋[/dim]")

if __name__ == "__main__":
    main()
