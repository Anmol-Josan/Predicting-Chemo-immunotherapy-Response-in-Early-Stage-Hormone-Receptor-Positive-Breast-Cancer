from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pypdfium2 as pdfium


SOFFICE = Path(r"C:\Program Files\LibreOffice\program\soffice.com")


def render(docx_path: Path, output_dir: Path, dpi: int = 144) -> Path:
    docx_path = docx_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not SOFFICE.exists():
        raise FileNotFoundError(f"LibreOffice was not found at {SOFFICE}")
    with tempfile.TemporaryDirectory(prefix="codex_lo_") as temporary:
        temporary_path = Path(temporary)
        profile = (temporary_path / "profile").as_uri()
        pdf_dir = temporary_path / "pdf"
        pdf_dir.mkdir()
        env = os.environ.copy()
        env["PATH"] = str(SOFFICE.parent) + os.pathsep + env.get("PATH", "")
        completed = subprocess.run(
            [
                str(SOFFICE),
                f"-env:UserInstallation={profile}",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(pdf_dir),
                str(docx_path),
            ],
            cwd=SOFFICE.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        pdf_path = pdf_dir / f"{docx_path.stem}.pdf"
        if completed.returncode != 0 or not pdf_path.exists():
            raise RuntimeError(f"LibreOffice conversion failed: {completed.stdout}\n{completed.stderr}")
        final_pdf = output_dir / pdf_path.name
        shutil.copy2(pdf_path, final_pdf)
    document = pdfium.PdfDocument(str(final_pdf))
    scale = dpi / 72
    for index, page in enumerate(document):
        image = page.render(scale=scale).to_pil()
        image.save(output_dir / f"page-{index + 1:03d}.png")
    return final_pdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dpi", type=int, default=144)
    arguments = parser.parse_args()
    print(render(arguments.docx, arguments.output_dir, arguments.dpi))
