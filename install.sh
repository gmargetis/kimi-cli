#!/usr/bin/env bash
set -e

echo "🤖 Installing Kimi CLI..."

# On macOS: ensure python3 is working (reinstall if broken)
if [[ "$(uname)" == "Darwin" ]]; then
    if ! python3 --version &>/dev/null 2>&1; then
        echo "🔧 python3 not working, installing via brew..."
        /opt/homebrew/bin/brew install python@3.13 2>/dev/null || brew install python@3.13
    elif [[ "$(wc -c < /opt/homebrew/Cellar/python@3.13/*/bin/python3 2>/dev/null | tr -d ' ')" == "0" ]]; then
        echo "🔧 python3 binary is empty, reinstalling via brew..."
        /opt/homebrew/bin/brew reinstall python@3.13
    fi
fi

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

# Create wrapper — kimi.py will auto-install deps on first run
cat > "$INSTALL_DIR/kimi" << 'WRAPPER'
#!/usr/bin/env bash
for PY in /opt/homebrew/bin/python3 /usr/local/bin/python3 python3 /usr/bin/python3; do
    if "$PY" --version &>/dev/null 2>&1; then
        exec "$PY" "$HOME/.kimi-cli/kimi.py" "$@"
    fi
done
echo "❌ python3 not found. Install it first: brew install python3"
exit 1
WRAPPER
chmod +x "$INSTALL_DIR/kimi"

# Check PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo ""
    echo "⚠️  Add to ~/.zshrc:"
    echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "   Then: source ~/.zshrc"
fi

echo ""
echo "✅ Done! First run will auto-install dependencies."
echo "   kimi \"your task\""
echo ""
echo "Set your API key:"
echo "   export NVIDIA_API_KEY=\"nvapi-...\""
