#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 test_rag_lookup_regression.py \
  "yellow rice" \
  "white sauce" \
  "red sauce" \
  "tomato wedges" \
  "diced onions" \
  "pickled red onion" \
  "pickled jalapeños" \
  "pickled turnips" \
  "baba ganoush" \
  "tzatziki" \
  "harissa sauce" \
  "roasted potatoes" \
  "roasted carrots" \
  "broccoli florets" \
  "green peas" \
  "Yorkshire puddings"
