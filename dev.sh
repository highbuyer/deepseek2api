#!/bin/bash
cd "$(dirname "$0")"
echo "=== deepseek2api dev server $(date) ==="
NODE_OPTIONS="--max-old-space-size=8192" DEBUG=1 npm start 2>&1 | tee /tmp/deepseek2api-live.log
