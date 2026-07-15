#!/usr/bin/env bash
set -euo pipefail

# Homepage performance budget.
#
# gzip budget (strict): bytes on the wire. 13,600 keeps the compressed page
# within roughly one TCP initial congestion window, so the homepage arrives
# in the first round trip.
#
# raw budget (loose): uncompressed bytes the browser parses. Caps unbounded
# growth without punishing well-compressed text content.

build_dir="${1:-public}"
html="$build_dir/index.html"
gzip_budget="${HOMEPAGE_GZIP_BUDGET_BYTES:-13600}"
raw_budget="${HOMEPAGE_RAW_BUDGET_BYTES:-28000}"

if [[ ! -f "$html" ]]; then
  echo "missing homepage: $html" >&2
  exit 1
fi

raw_bytes="$(wc -c < "$html" | tr -d ' ')"
gzip_bytes="$(gzip -9c "$html" | wc -c | tr -d ' ')"

echo "homepage raw bytes: $raw_bytes (budget: $raw_budget)"
echo "homepage gzip bytes: $gzip_bytes (budget: $gzip_budget)"

if (( raw_bytes > raw_budget )); then
  echo "homepage raw body exceeds budget" >&2
  exit 1
fi

if (( gzip_bytes > gzip_budget )); then
  echo "homepage gzip body exceeds budget" >&2
  exit 1
fi

if [[ -f "$build_dir/site.js" ]]; then
  js_raw="$(wc -c < "$build_dir/site.js" | tr -d ' ')"
  js_gzip="$(gzip -9c "$build_dir/site.js" | wc -c | tr -d ' ')"
  echo "deferred site.js raw bytes: $js_raw"
  echo "deferred site.js gzip bytes: $js_gzip"
fi
