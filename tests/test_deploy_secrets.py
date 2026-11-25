"""Local secret files must never be staged for upload to a Hugging Face Space."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_deploy_module():
    spec = importlib.util.spec_from_file_location("hf_space_deploy", ROOT / "deploy" / "huggingface_space.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage_skips_local_secret_files(tmp_path, monkeypatch):
    mod = _load_deploy_module()
    src = tmp_path / "repo"
    (src / ".streamlit").mkdir(parents=True)
    (src / ".streamlit" / "config.toml").write_text("[theme]\n")
    (src / ".streamlit" / "secrets.toml").write_text('HF_TOKEN = "not-a-real-token"\n')
    (src / "src").mkdir()
    (src / "src" / "x.py").write_text("")
    (src / "src" / ".env").write_text("KEY=not-a-real-key\n")
    (src / "src" / ".env.local").write_text("KEY=not-a-real-key\n")
    (src / "src" / "kaggle.json").write_text("{}")
    (src / "app.py").write_text("")
    monkeypatch.setattr(mod, "ROOT", src)

    dst = tmp_path / "staged"
    dst.mkdir()
    mod.stage(dst)

    staged = {p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file()}
    assert ".streamlit/config.toml" in staged
    assert "src/x.py" in staged
    assert "app.py" in staged
    for leaked in (".streamlit/secrets.toml", "src/.env", "src/.env.local", "src/kaggle.json"):
        assert leaked not in staged
