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
   > ⚠️ **Do not leave the default (3.14).** No TensorFlow release supports Python 3.14 yet (TensorFlow 2.21
   > supports 3.10–3.13), so the build stops with *"Error installing requirements"*. The version cannot be changed
   > after deployment: if this happens, delete the app (**My apps → ⋮ → Delete**) and create it again with 3.12.
   > The Python version box is a search field — clear any typed text, then click **3.12** in the list.
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
* If the build fails with a dependency error, check the Python version: 3.11, 3.12 and 3.13 were verified to
  resolve on Streamlit Cloud's Linux platform; 3.14 cannot work until TensorFlow publishes 3.14 wheels.
* `requirements.txt` installs the slimmer `tensorflow-cpu` on Linux and the standard `tensorflow` on Windows/macOS;
  both run on CPU.

## Hugging Face Spaces (fully scriptable, free)

A Docker Space gets 2 vCPU and 16 GB RAM on the free tier. The `Dockerfile` in this repository was built and run
locally as Hugging Face runs it (non-root user 1000, port 7860). All 13 pages, drawing, upload and batch prediction
passed browser tests, with about 0.6 GB of memory in use.

1. Create a free account at <https://huggingface.co/join>.
2. Create a token at <https://huggingface.co/settings/tokens>, type **Write** (or a fine-grained token with write
   access to your Spaces).
3. Deploy:
   ```bash
   pip install huggingface_hub
   ```
   ```bash
   set HF_TOKEN=hf_your_token_here          # Windows (macOS/Linux: export HF_TOKEN=...)
   ```
   ```bash
   python deploy/huggingface_space.py --space purf-ai-lab
   ```
4. The first build takes ~5–10 minutes. The app is then live at
   `https://huggingface.co/spaces/<username>/purf-ai-lab` (direct URL `https://<username>-purf-ai-lab.hf.space`).
   Re-run step 3 to publish updates. You can revoke the token afterwards.

## Any other Docker host (Render, Railway, Google Cloud Run, a VPS)
```bash
docker build -t purf .
```
```bash
docker run -p 7860:7860 purf
```
The container honours the `PORT` environment variable that these platforms set.
