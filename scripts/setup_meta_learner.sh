#!/usr/bin/env bash
set -euo pipefail

# 1. Create and source Virtual Environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch transformers flask pyyaml tqdm

# 3. Create directories
echo "Creating directories..."
mkdir -p meta_learner/data meta_learner/models

# 4. Start MetaLearner server
echo "Starting MetaLearner server..."
# We use nohup with the venv python
nohup python3 -m meta_learner.server > meta_learner/server.log 2>&1 &
PID=$!
echo "MetaLearner server started (PID $PID)"
echo "Backend is running on localhost:8001"
