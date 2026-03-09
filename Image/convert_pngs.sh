#!/bin/bash
set -e
cd "$(dirname "$0")"
# If run on the host ensure poppler-utils is available; when running inside a container
# the container should provide pdftoppm. Keep install steps if run on a clean linux host.
apt-get update -qq || true
apt-get install -y poppler-utils || true
mkdir -p pngs
for f in fig01_pipeline fig02_modeling fig03_roc fig04_architecture_bar fig05_ablation fig06_confusion fig07_shap fig08_dataset fig09_heatmap_umap diagram_caption; do
  if [ -f "$f.pdf" ]; then
    pdftoppm -png -r 600 "$f.pdf" "pngs/$f"
    if [ -f "pngs/${f}-1.png" ]; then mv "pngs/${f}-1.png" "pngs/${f}.png"; fi
  fi
done
ls -l pngs
