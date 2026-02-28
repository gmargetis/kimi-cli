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

# Find correct python3
if [[ "$(uname)" == "Darwin" ]]; then
    BREW_PREFIX="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
    PYTHON="$BREW_PREFIX/bin/python3"
    if [[ ! -x "$PYTHON" ]]; then
        PYTHON="$(which python3)"
    fi
else
    PYTHON="$(which python3)"
fi

echo "🐍 Using Python: $PYTHON"
"$PYTHON" -m pip install -r "$REPO_DIR/requirements.txt" -q

# Create wrapper with hardcoded python path
cat > "$INSTALL_DIR/kimi" << EOF
#!/usr/bin/env bash
exec "$PYTHON" "$REPO_DIR/kimi.py" "\$@"
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
