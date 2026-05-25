#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-public}"
html="$build_dir/index.html"
budget="${HOMEPAGE_BODY_BUDGET_BYTES:-13600}"

if [[ ! -f "$html" ]]; then
  echo "missing homepage: $html" >&2
  exit 1
fi

raw_bytes="$(wc -c < "$html" | tr -d ' ')"
gzip_bytes="$(gzip -9c "$html" | wc -c | tr -d ' ')"

echo "homepage raw bytes: $raw_bytes"
echo "homepage gzip bytes: $gzip_bytes"
echo "body budget bytes: $budget"

if (( raw_bytes > budget )); then
  echo "homepage raw body exceeds budget" >&2
  exit 1
fi

if (( gzip_bytes > budget )); then
  echo "homepage gzip body exceeds budget" >&2
  exit 1
fi

if [[ -f "$build_dir/site.js" ]]; then
  js_raw="$(wc -c < "$build_dir/site.js" | tr -d ' ')"
  js_gzip="$(gzip -9c "$build_dir/site.js" | wc -c | tr -d ' ')"
  echo "deferred site.js raw bytes: $js_raw"
  echo "deferred site.js gzip bytes: $js_gzip"
fi
