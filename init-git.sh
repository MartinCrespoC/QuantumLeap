#!/bin/bash

# Initialize Git repository and prepare for GitHub

echo "🔧 Initializing Git repository..."

# Initialize git if not already done
if [ ! -d ".git" ]; then
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already exists"
fi

# Add all files
git add .

# Show status
echo ""
echo "📋 Files to be committed:"
git status --short

echo ""
echo "📝 Ready to commit. Run:"
echo ""
echo "  git commit -m 'feat: QuantumLeap v0.4.0 - 801% faster LLM inference built on llama.cpp'"
echo "  git branch -M main"
echo "  git remote add origin https://github.com/YOUR_USERNAME/quantumleap.git"
echo "  git push -u origin main"
echo ""
echo "⚠️  Remember to:"
echo "  1. Create repository 'quantumleap' on GitHub first"
echo "  2. Replace YOUR_USERNAME in README.md badges (line 6)"
echo "  3. Update git remote URL above with your username"
echo ""
echo "🎯 Recommended repository name: quantumleap"
echo "📝 Description: ⚛️ 801% faster LLM inference built on llama.cpp"
echo ""
