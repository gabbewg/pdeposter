#!/bin/bash
# Renders poster_v2.html to a PDF (and a PNG preview) using headless Chrome.
# Works on Windows (Git Bash) and macOS.
# Usage: ./build_poster_v2.sh [input.html] [output.pdf]
#   Defaults: input = poster_v2.html, output = poster_v2.pdf

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT="${1:-"$SCRIPT_DIR/poster_v2.html"}"
OUTPUT="${2:-"${INPUT%.html}.pdf"}"

# Locate Chrome (Windows first, then macOS).
CHROME_CANDIDATES=(
    "/c/Program Files/Google/Chrome/Application/chrome.exe"
    "/c/Program Files (x86)/Google/Chrome/Application/chrome.exe"
    "$LOCALAPPDATA/Google/Chrome/Application/chrome.exe"
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)
CHROME=""
for c in "${CHROME_CANDIDATES[@]}"; do
    if [ -x "$c" ]; then CHROME="$c"; break; fi
done
if [ -z "$CHROME" ]; then
    echo "Google Chrome not found. Edit CHROME_CANDIDATES in this script."
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "Input not found: $INPUT"
    exit 1
fi

# Build a file:// URL Chrome accepts on this platform.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        # Convert /c/Users/... -> C:/Users/...
        WIN_INPUT="$(cygpath -m "$INPUT" 2>/dev/null || echo "$INPUT")"
        WIN_OUTPUT="$(cygpath -m "$OUTPUT" 2>/dev/null || echo "$OUTPUT")"
        URL="file:///$WIN_INPUT"
        OUT_ARG="$WIN_OUTPUT"
        ;;
    *)
        URL="file://$INPUT"
        OUT_ARG="$OUTPUT"
        ;;
esac

"$CHROME" \
    --headless \
    --disable-gpu \
    --no-pdf-header-footer \
    --no-margins \
    --print-to-pdf="$OUT_ARG" \
    "$URL" \
    2>/dev/null

echo "Created: $OUTPUT"

# Optional: render a PNG preview if pdftoppm is available
if command -v pdftoppm &>/dev/null; then
    PREVIEW="${OUTPUT%.pdf}_preview"
    pdftoppm -r 80 -png "$OUTPUT" "$PREVIEW"
    if [ -f "${PREVIEW}-1.png" ]; then
        mv "${PREVIEW}-1.png" "${PREVIEW}.png"
    fi
    echo "Preview: ${PREVIEW}.png"
fi
