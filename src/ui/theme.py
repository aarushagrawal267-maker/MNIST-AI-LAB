"""PURF visual design system: CSS, Plotly template and HTML building blocks.

Chrome (gradients, glows) uses the brand violet/cyan. Data marks use a
validated categorical palette (checked for colour-vision-deficiency
separation and >=3:1 contrast against the navy surface) so charts stay legible.
"""
from __future__ import annotations

import base64
import html
from typing import Iterable

import numpy as np

# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
BG = "#060913"
SURFACE = "#0f1631"
SURFACE_2 = "#141c3d"
BORDER = "rgba(148,163,255,0.14)"
TEXT = "#e8ecff"
TEXT_2 = "#b4bce3"
MUTED = "#7f89b8"
GRID = "rgba(148,163,255,0.08)"
AXIS = "rgba(148,163,255,0.22)"
BRAND = "#7c5cff"
BRAND_2 = "#22d3ee"
BRAND_3 = "#f472b6"

# Validated categorical order (dark steps) — assign in this order, never cycle.
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
# Sequential single-hue blue ramp (dark surface -> bright = more)
SEQ_BLUE = ["#0f1631", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"]
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
NEUTRAL = "#4b5583"

# Fixed colour per model (identity follows the entity, not its rank)
MODEL_COLORS = {
    "logistic_regression": SERIES[0],
    "knn": SERIES[1],
    "svm": SERIES[2],
    "ann": SERIES[3],
    "cnn_baseline": SERIES[4],
    "cnn_tuned": SERIES[5],
    "cnn_tuned_aug": SERIES[6],
}


def model_color(key: str) -> str:
    return MODEL_COLORS.get(key, SERIES[7])


# --------------------------------------------------------------------------- #
# Plotly
# --------------------------------------------------------------------------- #
def register_plotly_template() -> None:
    import plotly.graph_objects as go
    import plotly.io as pio

    if "purf" in pio.templates:
        pio.templates.default = "purf"
        return
    axis = dict(
        gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickcolor=AXIS,
        tickfont=dict(color=MUTED, size=12), title=dict(font=dict(color=TEXT_2, size=13)),
        showline=True, ticks="",
    )
    pio.templates["purf"] = go.layout.Template(
        layout=dict(
            font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif", color=TEXT, size=13),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=SERIES,
            xaxis=axis,
            yaxis=axis,
            margin=dict(l=48, r=18, t=48, b=44),
            title=dict(font=dict(family="'Space Grotesk', Inter, sans-serif", size=16, color=TEXT), x=0, xanchor="left"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_2), orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
            hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BRAND, font=dict(color=TEXT, family="Inter, sans-serif")),
            hovermode="closest",
            coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color=MUTED))),
            bargap=0.28,
        )
    )
    pio.templates.default = "purf"


def style_fig(fig, height: int | None = None, title: str | None = None, **layout):
    if height:
        fig.update_layout(height=height)
    if title is not None:
        fig.update_layout(title=title)
    if layout:
        fig.update_layout(**layout)
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    return fig


PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {{
  --bg: {BG}; --surface: {SURFACE}; --surface2: {SURFACE_2}; --border: {BORDER};
  --text: {TEXT}; --text2: {TEXT_2}; --muted: {MUTED};
  --brand: {BRAND}; --brand2: {BRAND_2}; --brand3: {BRAND_3};
  --radius: 18px;
}}

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {{
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}}
h1, h2, h3, h4, .purf-display {{ font-family: 'Space Grotesk', 'Inter', sans-serif !important; letter-spacing: -0.02em; }}

/* ---------- App background: deep space + aurora + grid ---------- */
.stApp {{
  background:
    radial-gradient(1200px 600px at 8% -10%, rgba(124,92,255,0.22), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,0.14), transparent 60%),
    radial-gradient(800px 600px at 50% 120%, rgba(244,114,182,0.10), transparent 60%),
    {BG};
  background-attachment: fixed;
}}
.stApp::before {{
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image: linear-gradient(rgba(148,163,255,0.045) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(148,163,255,0.045) 1px, transparent 1px);
  background-size: 44px 44px;
  mask-image: radial-gradient(ellipse at 50% 0%, black 0%, transparent 75%);
  -webkit-mask-image: radial-gradient(ellipse at 50% 0%, black 0%, transparent 75%);
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"], .block-container {{ padding-top: 2.2rem; max-width: 1320px; }}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(15,22,49,0.96), rgba(6,9,19,0.98));
  border-right: 1px solid var(--border);
}}
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] {{ border-radius: 10px; transition: all .15s ease; }}
[data-testid="stSidebarNavLink"]:hover {{ background: rgba(124,92,255,0.12); }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
  background: linear-gradient(90deg, rgba(124,92,255,0.28), rgba(34,211,238,0.08));
  box-shadow: inset 2px 0 0 var(--brand2);
}}
[data-testid="stSidebarNavSeparator"] {{ border-color: var(--border); }}
[data-testid="stNavSectionHeader"] {{ color: var(--muted) !important; letter-spacing: .14em; font-size: .68rem !important; text-transform: uppercase; }}

/* ---------- Widgets ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  border-radius: 12px; border: 1px solid var(--border); background: rgba(20,28,61,0.8);
  color: var(--text); font-weight: 600; transition: all .18s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: rgba(34,211,238,0.6); color: white; transform: translateY(-1px);
  box-shadow: 0 8px 24px rgba(34,211,238,0.14);
}}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"],
button[data-testid^="stBaseButton-primary"] {{
  background: linear-gradient(135deg, var(--brand), #4f46e5 50%, var(--brand2)); border: none; color: white;
  box-shadow: 0 8px 26px rgba(124,92,255,0.35);
}}
[data-testid="stPageLink"] a {{
  border: 1px solid var(--border); border-radius: 12px; background: rgba(20,28,61,0.6); padding: .45rem .8rem;
}}
[data-testid="stPageLink"] a:hover {{ border-color: var(--brand2); }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ border-radius: 10px 10px 0 0; padding: 8px 16px; color: var(--text2); }}
.stTabs [aria-selected="true"] {{ color: white !important; background: rgba(124,92,255,0.14); }}
.stTabs [data-baseweb="tab-highlight"] {{ background: linear-gradient(90deg, var(--brand), var(--brand2)); height: 2px; }}
[data-testid="stExpander"] {{ border: 1px solid var(--border); border-radius: 14px; background: rgba(15,22,49,0.55); }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border-radius: 14px; overflow: hidden; border: 1px solid var(--border); }}
[data-testid="stMetric"] {{
  background: rgba(15,22,49,0.6); border: 1px solid var(--border); border-radius: 16px; padding: 14px 16px;
}}
[data-testid="stMetricValue"] {{ font-family: 'Space Grotesk', sans-serif; }}
[data-testid="stFileUploaderDropzone"] {{
  border: 1.5px dashed rgba(124,92,255,0.45); background: rgba(15,22,49,0.55); border-radius: 16px;
}}
.stProgress > div > div > div > div {{ background: linear-gradient(90deg, var(--brand), var(--brand2)); }}
div[data-testid="stAlert"] {{ border-radius: 14px; }}
code {{ font-family: 'JetBrains Mono', monospace; }}
hr {{ border-color: var(--border); }}

/* ---------- Custom components ---------- */
.purf-hero {{
  position: relative; overflow: hidden; border-radius: 28px; padding: 46px 44px 40px;
  background: linear-gradient(140deg, rgba(124,92,255,0.20), rgba(15,22,49,0.75) 40%, rgba(34,211,238,0.10));
  border: 1px solid rgba(148,163,255,0.20);
  box-shadow: 0 30px 80px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
}}
.purf-hero::after {{
  content: ""; position: absolute; width: 520px; height: 520px; right: -140px; top: -220px; border-radius: 50%;
  background: radial-gradient(circle, rgba(34,211,238,0.30), transparent 65%); filter: blur(10px);
  animation: float 9s ease-in-out infinite;
}}
@keyframes float {{ 0%,100% {{ transform: translateY(0) }} 50% {{ transform: translateY(26px) }} }}
.purf-kicker {{
  display: inline-flex; align-items: center; gap: 8px; font-size: 12px; letter-spacing: .22em; text-transform: uppercase;
  color: var(--brand2); font-weight: 600; padding: 6px 12px; border: 1px solid rgba(34,211,238,0.35); border-radius: 999px;
  background: rgba(34,211,238,0.06);
}}
.purf-kicker .dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--brand2); box-shadow: 0 0 10px var(--brand2); animation: pulse 2s infinite; }}
@keyframes pulse {{ 0%,100% {{ opacity: 1 }} 50% {{ opacity: .35 }} }}
.purf-title {{
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: clamp(2.6rem, 6vw, 4.6rem); line-height: 1.02;
  margin: 18px 0 10px; letter-spacing: -0.04em;
  background: linear-gradient(92deg, #ffffff 10%, #c4b5fd 45%, var(--brand2) 85%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.purf-sub {{ color: var(--text2); font-size: 1.08rem; max-width: 760px; line-height: 1.6; }}
.grad-text {{
  background: linear-gradient(92deg, #c4b5fd, var(--brand2)); -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.purf-page-head {{ margin: 0 0 22px; }}
.purf-page-head .eyebrow {{ color: var(--brand2); font-size: 12px; letter-spacing: .22em; text-transform: uppercase; font-weight: 600; }}
.purf-page-head h1 {{ font-size: 2.35rem; margin: 6px 0 6px; padding: 0; font-weight: 700;
  background: linear-gradient(92deg, #fff, #c4b5fd 60%, var(--brand2)); -webkit-background-clip: text; background-clip: text; color: transparent; }}
.purf-page-head p {{ color: var(--text2); margin: 0; max-width: 860px; line-height: 1.55; }}

.purf-card {{
  position: relative; border-radius: var(--radius); padding: 20px 22px; height: 100%;
  background: linear-gradient(180deg, rgba(20,28,61,0.78), rgba(15,22,49,0.62));
  border: 1px solid var(--border); box-shadow: 0 16px 40px rgba(0,0,0,0.28);
  backdrop-filter: blur(10px); transition: transform .2s ease, border-color .2s ease;
}}
.purf-card:hover {{ transform: translateY(-2px); border-color: rgba(124,92,255,0.40); }}
.purf-card h4 {{ margin: 0 0 6px; font-size: 1.05rem; color: var(--text); }}
.purf-card p {{ margin: 0; color: var(--text2); font-size: .93rem; line-height: 1.55; }}
.purf-metric {{
  position: relative; border-radius: var(--radius); padding: 18px 20px 16px; overflow: hidden;
  background: linear-gradient(180deg, rgba(20,28,61,0.85), rgba(15,22,49,0.65)); border: 1px solid var(--border);
}}
.purf-metric::before {{ content: ""; position: absolute; left: 0; top: 0; height: 2px; width: 100%;
  background: linear-gradient(90deg, var(--accent, var(--brand)), transparent); }}
.purf-metric .label {{ color: var(--muted); font-size: .74rem; letter-spacing: .14em; text-transform: uppercase; font-weight: 600; }}
.purf-metric .value {{ font-family: 'Space Grotesk', sans-serif; font-size: clamp(1.35rem, 2.1vw, 2rem); font-weight: 700; color: var(--text); margin-top: 6px; line-height: 1.1; white-space: nowrap; }}
.purf-metric .sub {{ color: var(--text2); font-size: .84rem; margin-top: 4px; }}
.purf-metric .icon {{ position: absolute; right: 16px; top: 14px; font-size: 1.25rem; opacity: .8; }}

.purf-section {{ margin: 34px 0 14px; display: flex; align-items: baseline; gap: 14px; }}
.purf-section h3 {{ margin: 0; font-size: 1.35rem; }}
.purf-section .line {{ flex: 1; height: 1px; background: linear-gradient(90deg, var(--border), transparent); }}
.purf-section .note {{ color: var(--muted); font-size: .85rem; }}

.badge {{ display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font-size: .75rem;
  font-weight: 600; border: 1px solid var(--border); color: var(--text2); background: rgba(20,28,61,0.7); }}
.badge.trax {{ color: white; border: none; background: linear-gradient(135deg, var(--brand), var(--brand2)); box-shadow: 0 0 18px rgba(124,92,255,.45); }}
.badge.good {{ color: #b8f5b8; border-color: rgba(12,163,12,.5); }}
.badge.bad {{ color: #ffc0c0; border-color: rgba(208,59,59,.6); }}

/* Prediction orb */
.purf-orb-wrap {{ display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 8px 0 4px; }}
.purf-orb {{
  position: relative; width: 210px; height: 210px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  background: radial-gradient(circle at 35% 30%, rgba(124,92,255,0.55), rgba(15,22,49,0.95) 62%);
  box-shadow: 0 0 0 1px rgba(148,163,255,0.25), 0 0 60px rgba(124,92,255,0.45), 0 0 120px rgba(34,211,238,0.18), inset 0 0 40px rgba(34,211,238,0.18);
  animation: orbIn .55s cubic-bezier(.2,.9,.3,1.3);
}}
.purf-orb::before {{
  content: ""; position: absolute; inset: -8px; border-radius: 50%; padding: 2px;
  background: conic-gradient(var(--brand2) calc(var(--p) * 1%), rgba(148,163,255,0.12) 0);
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude;
}}
@keyframes orbIn {{ from {{ transform: scale(.6); opacity: 0 }} to {{ transform: scale(1); opacity: 1 }} }}
.purf-orb .digit {{ font-family: 'Space Grotesk', sans-serif; font-size: 128px; font-weight: 700; line-height: 1; color: white;
  text-shadow: 0 0 30px rgba(34,211,238,0.8), 0 0 60px rgba(124,92,255,0.6); }}
.purf-orb-caption {{ margin-top: 18px; text-align: center; }}
.purf-orb-caption .conf {{ font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 700; color: white; }}
.purf-orb-caption .lbl {{ color: var(--muted); font-size: .78rem; letter-spacing: .18em; text-transform: uppercase; }}

.purf-top3 {{ display: flex; flex-direction: column; gap: 10px; }}
.purf-top3 .row {{ display: grid; grid-template-columns: 34px 1fr 64px; gap: 12px; align-items: center; }}
.purf-top3 .d {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; text-align: center; color: var(--text); }}
.purf-top3 .bar {{ height: 10px; border-radius: 6px; background: rgba(148,163,255,0.10); overflow: hidden; }}
.purf-top3 .bar span {{ display: block; height: 100%; border-radius: 6px; background: linear-gradient(90deg, var(--brand), var(--brand2)); }}
.purf-top3 .row:not(:first-child) .bar span {{ background: #4b5583; }}
.purf-top3 .p {{ font-family: 'JetBrains Mono', monospace; font-size: .9rem; color: var(--text2); text-align: right; }}

.purf-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
.purf-step {{ border: 1px solid var(--border); border-radius: 16px; padding: 14px; background: rgba(15,22,49,0.55); }}
.purf-step .n {{ font-family: 'JetBrains Mono', monospace; color: var(--brand2); font-size: .78rem; }}
.purf-step .t {{ font-weight: 600; margin-top: 4px; }}
.purf-step .d {{ color: var(--muted); font-size: .82rem; margin-top: 4px; line-height: 1.45; }}

.purf-img-tile {{ border: 1px solid var(--border); border-radius: 14px; padding: 8px; background: rgba(15,22,49,0.6); text-align: center; }}
.purf-img-tile img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #000; image-rendering: pixelated; border-radius: 8px; display: block; }}
.purf-img-tile .cap {{ font-size: .78rem; color: var(--text2); margin-top: 6px; line-height: 1.35; }}
.purf-img-tile.bad {{ border-color: rgba(208,59,59,.45); }}

.purf-footer {{ margin-top: 48px; padding: 18px 0 6px; border-top: 1px solid var(--border); color: var(--muted); font-size: .82rem;
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }}
.purf-brand {{ display: flex; align-items: center; gap: 12px; padding: 6px 4px 14px; }}
.purf-brand .logo {{ width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center;
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: white; font-size: 20px;
  background: linear-gradient(135deg, var(--brand), var(--brand2)); box-shadow: 0 0 22px rgba(124,92,255,.55); }}
.purf-brand .name {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.25rem; letter-spacing: .06em; color: white; }}
.purf-brand .tag {{ color: var(--muted); font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; }}
</style>
"""


def inject_css() -> None:
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
def esc(s) -> str:
    return html.escape(str(s))


def png_data_uri(img: np.ndarray, upscale: int = 1) -> str:
    from ..image_processor import image_to_png_bytes

    return "data:image/png;base64," + base64.b64encode(image_to_png_bytes(img, upscale)).decode()


def rgb_data_uri(rgb: np.ndarray, upscale: int = 1) -> str:
    import cv2

    a = np.asarray(rgb, dtype=np.uint8)
    if upscale > 1:
        a = cv2.resize(a, (a.shape[1] * upscale, a.shape[0] * upscale), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    import streamlit as st

    st.markdown(
        f'<div class="purf-page-head"><div class="eyebrow">{esc(eyebrow)}</div>'
        f"<h1>{esc(title)}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def section(title: str, note: str = "") -> None:
    import streamlit as st

    st.markdown(
        f'<div class="purf-section"><h3>{esc(title)}</h3><div class="line"></div>'
        f'<div class="note">{esc(note)}</div></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str = "", icon: str = "", accent: str = BRAND) -> str:
    return (
        f'<div class="purf-metric" style="--accent:{accent}"><div class="icon">{icon}</div>'
        f'<div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def metric_row(cards: Iterable[str]) -> None:
    import streamlit as st

    cards = list(cards)
    cols = st.columns(len(cards))
    for c, html_card in zip(cols, cards):
        c.markdown(html_card, unsafe_allow_html=True)


def card(title: str, body: str, icon: str = "") -> str:
    return f'<div class="purf-card"><h4>{icon} {esc(title)}</h4><p>{body}</p></div>'


def badge(text: str, kind: str = "") -> str:
    return f'<span class="badge {kind}">{esc(text)}</span>'


def image_tile(img: np.ndarray, caption: str = "", upscale: int = 4, bad: bool = False, rgb: bool = False) -> str:
    uri = rgb_data_uri(img, upscale) if rgb else png_data_uri(img, upscale)
    return (f'<div class="purf-img-tile{" bad" if bad else ""}"><img src="{uri}"/>'
            f'<div class="cap">{caption}</div></div>')


def image_grid(tiles: list[str], columns: int = 6) -> None:
    import streamlit as st

    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat({columns},minmax(0,1fr));gap:10px">'
        + "".join(tiles) + "</div>",
        unsafe_allow_html=True,
    )


def prediction_orb(digit: int, confidence: float, label: str = "confidence") -> str:
    return (
        f'<div class="purf-orb-wrap"><div class="purf-orb" style="--p:{confidence * 100:.1f}">'
        f'<div class="digit">{digit}</div></div>'
        f'<div class="purf-orb-caption"><div class="conf">{confidence * 100:.1f}%</div>'
        f'<div class="lbl">{esc(label)}</div></div></div>'
    )


def top3_bars(top3: list[tuple[int, float]]) -> str:
    rows = "".join(
        f'<div class="row"><div class="d">{d}</div><div class="bar"><span style="width:{max(p * 100, 0.6):.1f}%"></span></div>'
        f'<div class="p">{p * 100:.2f}%</div></div>'
        for d, p in top3
    )
    return f'<div class="purf-top3">{rows}</div>'


def footer() -> None:
    import streamlit as st

    from ..config import LAB_NAME, MODEL_NAME

    st.markdown(
        f'<div class="purf-footer"><span>{LAB_NAME} · handwritten digit intelligence lab · powered by {MODEL_NAME}</span>'
        f"<span>MNIST via Kaggle (hojjatk/mnist-dataset) · TensorFlow · scikit-learn · Streamlit</span></div>",
        unsafe_allow_html=True,
    )
