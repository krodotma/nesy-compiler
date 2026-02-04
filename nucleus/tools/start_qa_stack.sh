#!/bin/bash
# Start the Pluribus QA stack via PM2 (observer + tool queue).

set -e

cd "$(dirname "$0")"

if ! command -v pm2 &> /dev/null; then
  echo "PM2 not found. Installing via npm..."
  npm install -g pm2
fi

echo "Starting Pluribus QA stack..."
pm2 start ecosystem.qa_stack.config.cjs

echo "---------------------------------------------------"
echo "✅ QA Stack Online"
echo "🔎 qa-observer + qa-tool-queue + qa-action-executor running under PM2"
echo "---------------------------------------------------"
echo "Use 'pm2 status' to monitor processes."
echo "Use 'pm2 logs qa-observer' for observer output."
echo "Use 'pm2 logs qa-tool-queue' for queue output."
echo "Use 'pm2 logs qa-action-executor' for action execution output."
