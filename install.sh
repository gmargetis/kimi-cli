#!/usr/bin/env bash
set -e

echo "🤖 Installing Kimi CLI..."

INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR"

REPO_DIR="$HOME/.kimi-cli"
if [ -d "$REPO_DIR" ]; then
    echo "📦 Updating..."
    git -C "$REPO_DIR" pull
else
    echo "📦 Cloning..."
    git clone https://github.com/gmargetis/kimi-cli.git "$REPO_DIR"
fi

# Install dependencies using the same python3 that's in PATH
echo "📦 Installing Python dependencies..."
python3 -m pip install -r "$REPO_DIR/requirements.txt" -q 2>/dev/null || \
    pip3 install -r "$REPO_DIR/requirements.txt" -q 2>/dev/null || \
    /opt/homebrew/bin/python3 -m pip install -r "$REPO_DIR/requirements.txt" -q

# Create wrapper — use env to find python3 at runtime (avoids symlink issues)
cat > "$INSTALL_DIR/kimi" << 'EOF'
#!/bin/sh
SCRIPT="$HOME/.kimi-cli/kimi.py"
# Try python3 from common locations
for PY in python3 /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
    if command -v "$PY" &>/dev/null 2>&1; then
        exec "$PY" "$SCRIPT" "$@"
    fi
done
echo "❌ python3 not found"
exit 1
EOF
chmod +x "$INSTALL_DIR/kimi"

# Check PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo ""
    echo "⚠️  Add this to your ~/.zshrc or ~/.bashrc:"
    echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "   Then run: source ~/.zshrc"
fi

echo ""
echo "✅ Done! Usage:"
echo "   kimi \"your task\""
echo "   kimi -w ~/project"
echo ""
echo "Set your API key:"
echo "   export NVIDIA_API_KEY=\"nvapi-...\""
