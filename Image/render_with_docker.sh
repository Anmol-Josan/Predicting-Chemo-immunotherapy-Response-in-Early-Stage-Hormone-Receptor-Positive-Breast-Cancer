#!/bin/bash
set -e

# Run from this script's directory and build all .tex figures using a LaTeX Docker image.
cd "$(dirname "$0")"

echo "Rendering LaTeX figures in $(pwd) using Docker (blang/latex:ctanfull)"

for f in *.tex; do
  echo "-- Rendering $f"
  docker run --rm -v "$(pwd):/work" -w /work blang/latex:ctanfull \
    pdflatex -interaction=nonstopmode -halt-on-error "$f"
  # second pass to settle cross-references if needed
  docker run --rm -v "$(pwd):/work" -w /work blang/latex:ctanfull \
    pdflatex -interaction=nonstopmode -halt-on-error "$f" || true
done

echo "Converting PDFs to PNGs"
./convert_pngs.sh

echo "Done. Generated PNGs are in the 'pngs' subdirectory."
