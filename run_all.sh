#!/bin/bash
# ============================================================
# NLP Assignment 2 Pipeline Runner (Task 1 -> Task 2 -> Task 3)
# ============================================================

set -e
cd "$(dirname "$0")"

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

PIPELINE_START=$(date +%s)
echo "============================================================"
echo "🚀 Starting NLP Assignment 2 Pipeline at $(date)"
echo "============================================================"

# --- Task 1 ---
echo -e "\n>>> Starting Task 1 (Probability Detection)..."
python -u task1.py

# --- Task 2 ---
echo -e "\n>>> Starting Task 2 (SelfCheckGPT)..."
python -u task2.py

# --- Task 3 ---
echo -e "\n>>> Starting Task 3 (Stacked Ensemble)..."
python -u task3.py

PIPELINE_END=$(date +%s)
TOTAL_DIFF=$((PIPELINE_END - PIPELINE_START))
HOURS=$((TOTAL_DIFF / 3600))
MINS=$(((TOTAL_DIFF % 3600) / 60))
SECS=$((TOTAL_DIFF % 60))

echo -e "\n============================================================"
echo "🎉 ALL TASKS FINISHED SUCCESSFULLY!"
echo "⏱ Total Pipeline Runtime: ${HOURS}h ${MINS}m ${SECS}s (${TOTAL_DIFF} seconds)"
echo "============================================================"

