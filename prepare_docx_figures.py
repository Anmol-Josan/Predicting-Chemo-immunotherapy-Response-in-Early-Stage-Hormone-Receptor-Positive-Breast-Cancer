from pathlib import Path

from PIL import Image


root = Path(__file__).resolve().parent
source = root / "results"
target = root / "_docx_figures"
target.mkdir(exist_ok=True)
names = [
    "pca_explained_variance.png",
    "patient_prediction_summary.png",
    "loss_curves_MLP.png",
    "loss_curves_CNN.png",
    "loss_curves_BiLSTM.png",
    "loss_curves_Transformer.png",
    "loss_curves_XGBoost.png",
    "shap_summary_plot.png",
]
for name in names:
    with Image.open(source / name) as image:
        converted = image.convert("RGB")
        converted.thumbnail((1800, 2200), Image.Resampling.LANCZOS)
        converted.save(target / name, "PNG", optimize=True, dpi=(150, 150))
