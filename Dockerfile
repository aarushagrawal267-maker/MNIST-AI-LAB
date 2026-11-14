# PURF lab — container image (Hugging Face Spaces, Render, Railway, Cloud Run, any Docker host)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PURF_PUBLIC_DEMO=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=user . .
USER user

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s CMD curl -fs http://localhost:${PORT}/_stcore/health || exit 1

# XSRF/CORS are relaxed because hosting platforms embed the app in an iframe on another origin;
# without this, file uploads are rejected with HTTP 403.
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --browser.gatherUsageStats=false"]
