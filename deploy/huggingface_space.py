"""Deploy PURF to a Hugging Face Space (Docker SDK).

    HF_TOKEN=hf_xxx python deploy/huggingface_space.py --space purf-ai-lab

Creates (or updates) https://huggingface.co/spaces/<user>/<space> and uploads the app.
The token needs write access; it is read from the HF_TOKEN environment variable and never stored.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ["app.py", "config.yaml", "requirements.txt", "Dockerfile", ".dockerignore", "src", "pages", "assets",
           ".streamlit", "models", "outputs", "data/raw/.gitkeep", "data/processed/.gitkeep"]
EXCLUDE_PARTS = {"__pycache__", "custom", ".ipynb_checkpoints"}

SPACE_README = """---
title: PURF AI Lab
emoji: ✍️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Draw a digit and watch TRAX, a 99.5%-accurate CNN, read it
---

# PURF — Handwritten Digit Intelligence Lab

Draw a digit and watch **TRAX**, a convolutional neural network trained on MNIST
(Kaggle `hojjatk/mnist-dataset`, 99.51% test accuracy), recognise it. Explore the dataset, compare five model
families, analyse errors and look inside the network with explainable AI.

Source code: https://github.com/aarushagrawal267-maker/MNIST-AI-LAB
"""


def stage(dst: Path) -> None:
    for item in INCLUDE:
        src = ROOT / item
        if not src.exists():
            continue
        target = dst / item
        if src.is_dir():
            shutil.copytree(src, target, ignore=lambda d, names: [n for n in names if n in EXCLUDE_PARTS or n.endswith(".pyc")])
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    (dst / "README.md").write_text(SPACE_README, encoding="utf-8")


def main() -> None:
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="purf-ai-lab", help="Space name (URL slug)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to a Hugging Face token with write access.")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.space}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="space",
                          commit_message="Deploy PURF lab", delete_patterns=["*"])
    print(f"Deployed: https://huggingface.co/spaces/{repo_id}")
    print(f"Direct app URL: https://{user.lower()}-{args.space.lower()}.hf.space")


if __name__ == "__main__":
    main()
