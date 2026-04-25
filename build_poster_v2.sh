#!/bin/bash
# Renders poster_v2.html to a A1 PDF (and a PNG preview) using headless Chrome.
# Usage: ./build_poster_v2.sh [input.html] [output.pdf]
#   Defaults: input = poster_v2.html, output = poster_v2.pdf

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT="${1:-"$SCRIPT_DIR/poster_v2.html"}"
OUTPUT="${2:-"${INPUT%.html}.pdf"}"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ ! -x "$CHROME" ]; then
    echo "Google Chrome not found at: $CHROME"
    echo "Install Chrome, or edit the CHROME path in this script."
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "Input not found: $INPUT"
    exit 1
fi

"$CHROME" \
    --headless \
    --disable-gpu \
    --no-pdf-header-footer \
    --no-margins \
    --print-to-pdf="$OUTPUT" \
    "file://$INPUT" \
    2>/dev/null

echo "Created: $OUTPUT"

# Optional: render a PNG preview if pdftoppm is available
if command -v pdftoppm &>/dev/null; then
    PREVIEW="${OUTPUT%.pdf}_preview"
    pdftoppm -r 80 -png "$OUTPUT" "$PREVIEW"
    # pdftoppm appends "-1" for the first page; rename to a stable filename
    if [ -f "${PREVIEW}-1.png" ]; then
        mv "${PREVIEW}-1.png" "${PREVIEW}.png"
    fi
    echo "Preview: ${PREVIEW}.png"
fi
