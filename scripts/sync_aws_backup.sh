#!/usr/bin/env bash
set -euo pipefail

# Syncs a consistent SQLite + ChromaDB snapshot from AWS EC2 to your local Mac.
AWS_HOST="${AWS_HOST:-ubuntu@16.178.131.45}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/aws-proxy-key.pem}"
PROJECT_DIR="/Users/siddhipatel/.gemini/antigravity-ide/scratch/personal-ai-proxy"
BACKUP_DEST="${BACKUP_DEST:-$PROJECT_DIR/backups/latest_data}"
REMOTE_DIR="personal-ai-proxy"
REMOTE_STAGE="/tmp/proxy_backup_stage"

mkdir -p "$BACKUP_DEST"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating safe hot snapshot on AWS ($AWS_HOST)..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "$AWS_HOST" \
  "rm -rf $REMOTE_STAGE && mkdir -p $REMOTE_STAGE && cd ~/$REMOTE_DIR && ./venv/bin/python -c \"from src.core.backup_manager import _stage_data_snapshot; _stage_data_snapshot('$REMOTE_STAGE')\""

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Downloading snapshot to $BACKUP_DEST..."
rsync -az --delete -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new" \
  "$AWS_HOST:$REMOTE_STAGE/" "$BACKUP_DEST/"

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$AWS_HOST" "rm -rf $REMOTE_STAGE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Backup completed cleanly at $BACKUP_DEST"
