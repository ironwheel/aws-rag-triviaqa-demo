#!/bin/bash
# @file install.sh
# @brief Environment setup script for aws-rag-triviaqa-demo
# @author Robert E. Taylor
# @license MIT

set -e

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete."

