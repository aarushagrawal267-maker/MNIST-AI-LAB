# Deploying PURF to a public URL

The app is ready for **Streamlit Community Cloud** (free, permanent URL, deploys straight from GitHub).
This exact setup was rehearsed before release: fresh clone in `/mount/src/<repo>` (the path Streamlit Cloud uses),
clean Python 3.12 environment installed with `uv` from `requirements.txt`, no cached dataset. Result: all 13 pages
loaded without errors, MNIST downloaded from Kaggle on first use, drawn digits were recognised, and the server used
about 1 GB of memory.

## Streamlit Community Cloud (recommended, ~3 minutes)

1. Open <https://share.streamlit.io> and click **Continue with GitHub**. Sign in as the owner of
   `aarushagrawal267-maker/MNIST-AI-LAB` and authorise Streamlit.
2. Click **Create app** → **Deploy a public app from GitHub** (*"Yup, I have an app"*).
3. Fill in:
   | Field | Value |
   |---|---|
   | Repository | `aarushagrawal267-maker/MNIST-AI-LAB` |
   | Branch | `claude/keen-hamilton-qdlk3u` (the repository's default branch) |
   | Main file path | `app.py` |
   | App URL (optional) | e.g. `purf-ai-lab` → `https://purf-ai-lab.streamlit.app` |
4. Open **Advanced settings** → **Python version: 3.12** → **Save**. No secrets are needed.
5. Click **Deploy**. The first build installs TensorFlow and takes about 3–6 minutes. The first visit then downloads
   MNIST from Kaggle (~20 s); after that, pages load from cache.

### What visitors can do
Everything — draw, upload, batch-predict, explore data, compare and evaluate models, error analysis and explainable AI.

### Public demo mode (automatic)
On Streamlit Cloud the app detects that it is shared by everyone and switches on *public demo mode*:
* in-app training is limited to 10,000 images and 5 epochs, one job at a time per server;
* models trained by visitors are temporary and cannot replace TRAX or be deleted by visitors.

To enable the same limits elsewhere, set the environment variable `PURF_PUBLIC_DEMO=1`.

### Updating the live app
Push to the deployed branch — Streamlit Cloud redeploys automatically.

### Notes
* Free apps sleep after a period without visitors; the first visitor then waits ~30 s for it to wake.
* If the build fails with a dependency error, check that Python 3.12 (or 3.11) is selected — TensorFlow 2.21 has
  no wheels for Python 3.14.
* `requirements.txt` installs the slimmer `tensorflow-cpu` on Linux and the standard `tensorflow` on Windows/macOS;
  both run on CPU.
