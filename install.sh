#!/usr/bin/env bash
set -e

echo "🤖 Installing Kimi CLI..."

# Install to ~/.local/bin
INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR"

# Clone or update
REPO_DIR="$HOME/.kimi-cli"
if [ -d "$REPO_DIR" ]; then
    echo "📦 Updating..."
    git -C "$REPO_DIR" pull
else
    echo "📦 Cloning..."
    git clone https://github.com/gmargetis/kimi-cli.git "$REPO_DIR"
fi

# Install dependencies — use brew python on macOS if available
if [[ "$(uname)" == "Darwin" ]]; then
    PYTHON="$(brew --prefix 2>/dev/null)/bin/python3"
    if [[ ! -x "$PYTHON" ]]; then
        PYTHON="python3"
    fi
else
    PYTHON="python3"
fi

"$PYTHON" -m pip install -r "$REPO_DIR/requirements.txt" -q

# Create wrapper script
cat > "$INSTALL_DIR/kimi" << WRAPPER
#!/usr/bin/env bash
exec "$PYTHON" "\$HOME/.kimi-cli/kimi.py" "\$@"
WRAPPER
chmod +x "$INSTALL_DIR/kimi"

# Check PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo ""
    echo "⚠️  Add this to your ~/.bashrc or ~/.zshrc:"
    echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

echo ""
echo "✅ Done! Usage:"
echo "   kimi \"your task\""
echo "   kimi -w ~/project"
echo ""
echo "Set your API key:"
echo "   export NVIDIA_API_KEY=\"nvapi-...\""
