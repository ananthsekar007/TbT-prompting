#!/bin/bash
# Builds A4_1237228176.zip without the benchmark data, the API key or the LLM cache.
set -e
cd "$(dirname "$0")"
rm -f A4_1237228176.zip
zip -rq A4_1237228176.zip . \
  -x ".env" "benchmarks/*" "cache/*" "*__pycache__*" "*.DS_Store" "A4_*.zip" "A4_*.pdf" ".git/*"
echo "wrote A4_1237228176.zip ($(du -h A4_1237228176.zip | cut -f1))"
unzip -l A4_1237228176.zip | grep -E "\.env$|training_0|test_0|target\.csv" && echo "WARNING: forbidden file in zip" || true
