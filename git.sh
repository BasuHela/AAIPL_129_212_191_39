#!/bin/bash
# https://chatgpt.com/share/687b497d-908c-8008-b0fd-3b67d22800b2
set -e  # Stop script on error

# ====== USER CONFIGURATION ======
REPO_NAME="AAIPL_129_212_191_39"      # <-- Change this to actual IP
VISIBILITY="private"                    # or "private"
COMMIT_MESSAGE="Initial commit"
GITHUB_USERNAME="Basu"        # <-- Change this
GITHUB_EMAIL="basuhela@example.com"         # <-- Change this
# =================================

# Step 1: Install GitHub CLI if missing
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) not found. Installing..."
    sudo apt update
    sudo apt install -y curl git
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
      gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
      sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt update
    sudo apt install -y gh
fi

# Step 2: Authenticate GitHub CLI
if ! gh auth status &>/dev/null; then
    echo "Please authenticate with GitHub CLI:"
    gh auth login
fi

# Step 3: Create GitHub Repo
echo "Creating remote repo on GitHub..."
gh repo create "$REPO_NAME" --$VISIBILITY --confirm

# Step 4: Git setup
echo "Initializing local Git repo..."
git init
git config --global credential.helper store
git config --global user.name "$GITHUB_USERNAME"
git config --global user.email "$GITHUB_EMAIL"

git add .
git commit -m "$COMMIT_MESSAGE"
git branch -M main

# Step 5: Set remote & push
REMOTE_URL="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
git remote add origin "$REMOTE_URL"

echo "Pushing code to GitHub..."
git push -u origin main

echo "✅ Done! Repo pushed to: $REMOTE_URL"
