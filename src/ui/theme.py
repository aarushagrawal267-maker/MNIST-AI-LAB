"""PURF visual design system: CSS, Plotly template and HTML building blocks.

Chrome is a quiet graphite ramp with one pencil-blue accent. Data marks use a
validated categorical palette (checked for colour-vision-deficiency
separation and >=3:1 contrast against a dark surface) so charts stay legible.
"""
from __future__ import annotations

import base64
import html
from typing import Iterable

import numpy as np

# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
# Graphite ramp (a pencil on paper, dark) plus one pencil-blue accent.
BG = "#141415"
SIDEBAR = "#0f0f10"
SURFACE = "#19191b"
SURFACE_2 = "#232325"
BORDER = "rgba(237,237,237,0.08)"
BORDER_STRONG = "rgba(237,237,237,0.14)"
TEXT = "#ededed"
TEXT_2 = "#a1a1a6"
MUTED = "#85858b"
GRID = "rgba(237,237,237,0.06)"
AXIS = "rgba(237,237,237,0.16)"
ACCENT = "#7fa7ee"
# Legacy names kept so page code keeps working; all point at the one accent.
BRAND = ACCENT
BRAND_2 = ACCENT
BRAND_3 = ACCENT

# Validated categorical order (dark steps) — assign in this order, never cycle.
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
# Sequential single-hue blue ramp (card surface -> bright = more)
SEQ_BLUE = ["#19191b", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"]
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
NEUTRAL = "#4a4a50"

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
        tickfont=dict(color=MUTED, size=12, family="'IBM Plex Mono', monospace"), title=dict(font=dict(color=TEXT_2, size=13)),
        showline=True, ticks="",
    )
    pio.templates["purf"] = go.layout.Template(
        layout=dict(
            font=dict(family="'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif", color=TEXT, size=13),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=SERIES,
            xaxis=axis,
            yaxis=axis,
            margin=dict(l=48, r=20, t=48, b=44),
            title=dict(font=dict(family="'IBM Plex Sans', sans-serif", size=15, color=TEXT), x=0, xanchor="left"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_2), orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
            hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=AXIS, font=dict(color=TEXT, family="'IBM Plex Mono', monospace")),
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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {{
  --bg: {BG}; --sidebar: {SIDEBAR}; --surface: {SURFACE}; --surface2: {SURFACE_2};
  --border: {BORDER}; --border-strong: {BORDER_STRONG};
  --text: {TEXT}; --text2: {TEXT_2}; --muted: {MUTED}; --accent: {ACCENT};
  --brand: {ACCENT}; --brand2: {ACCENT};
  --radius: 14px; --radius-sm: 8px;
  --ease: cubic-bezier(0.22, 1, 0.36, 1);
  --mono: 'IBM Plex Mono', ui-monospace, monospace;
}}

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {{
  font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
}}
h1, h2, h3, h4, .purf-display {{ font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 500 !important; letter-spacing: 0; }}

/* ---------- App background: plain graphite ---------- */
.stApp {{ background: {BG}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"], .block-container {{ padding-top: 40px; max-width: 1440px; }}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{ background: var(--sidebar); border-right: 1px solid var(--border); }}
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] {{
  border-radius: var(--radius-sm); transition: background-color .15s var(--ease), color .15s var(--ease);
}}
[data-testid="stSidebarNavLink"]:hover {{ background: var(--surface); }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
  background: var(--surface2); box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
}}
[data-testid="stSidebarNavSeparator"] {{ border-color: var(--border); }}
[data-testid="stNavSectionHeader"] {{ color: var(--muted) !important; letter-spacing: 0; font-size: 12px !important; text-transform: none; font-weight: 500; }}

/* ---------- Widgets ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  border-radius: var(--radius-sm); border: 1px solid var(--border-strong); background: var(--surface2);
  color: var(--text); font-weight: 500;
  transition: background-color .15s var(--ease), border-color .15s var(--ease), color .15s var(--ease), transform .12s var(--ease);
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: rgba(237,237,237,0.28); color: var(--text); background: #2a2a2d; }}
.stButton > button:active, .stDownloadButton > button:active {{ transform: scale(0.98); }}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible, [data-testid="stPageLink"] a:focus-visible {{
  outline: 2px solid var(--accent); outline-offset: 2px;
}}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"],
button[data-testid^="stBaseButton-primary"] {{
  background: var(--text); border: 1px solid var(--text); color: #141415;
}}
button[data-testid^="stBaseButton-primary"]:hover {{ background: #ffffff; color: #141415; }}
[data-testid="stPageLink"] a {{
  border: 1px solid var(--border-strong); border-radius: var(--radius-sm); background: var(--surface); padding: 8px 12px;
  transition: background-color .15s var(--ease), border-color .15s var(--ease);
}}
[data-testid="stPageLink"] a:hover {{ border-color: rgba(237,237,237,0.28); background: var(--surface2); }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ border-radius: var(--radius-sm) var(--radius-sm) 0 0; padding: 8px 16px; color: var(--text2); }}
.stTabs [aria-selected="true"] {{ color: var(--text) !important; background: transparent; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--accent); height: 2px; }}
[data-testid="stExpander"] {{ border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border-radius: var(--radius); overflow: hidden; border: 1px solid var(--border); }}
[data-testid="stMetric"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 20px; }}
[data-testid="stMetricValue"] {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
[data-testid="stFileUploaderDropzone"] {{ border: 1px dashed var(--border-strong); background: var(--surface); border-radius: var(--radius); }}
.stProgress > div > div > div > div {{ background: var(--accent); }}
div[data-testid="stAlert"] {{ border-radius: var(--radius); }}
code {{ font-family: var(--mono); color: var(--text2) !important; background: var(--surface2) !important; }}
hr {{ border-color: var(--border); }}

/* ---------- Hero ---------- */
.purf-hero {{
  display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 48px; align-items: center;
  padding: 24px 0 32px; border-bottom: 1px solid var(--border);
}}
.purf-kicker {{ display: inline-flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text2); font-family: var(--mono); }}
.purf-kicker .dot {{ width: 8px; height: 8px; background: var(--accent); }}
.purf-kicker .dot.off {{ background: var(--muted); }}
.purf-title {{
  font-family: 'IBM Plex Sans', sans-serif; font-weight: 500; font-size: clamp(2.5rem, 5vw, 4rem); line-height: 1.06;
  margin: 16px 0 16px; letter-spacing: -0.03em; color: var(--text);
}}
.purf-sub {{ color: var(--text2); font-size: 18px; max-width: 40rem; line-height: 1.6; }}
.purf-sub b {{ color: var(--text); font-weight: 500; }}
.purf-specimen {{ display: flex; align-items: center; gap: 24px; padding: 24px; border: 1px solid var(--border);
  border-radius: 20px; background: var(--surface); }}
.purf-specimen .px {{ width: 168px; height: 168px; image-rendering: pixelated; display: block; border-radius: 4px;
  background: #000; box-shadow: 0 0 0 1px var(--border); }}
.purf-specimen .read {{ display: flex; flex-direction: column; gap: 4px; min-width: 112px; }}
.purf-specimen .d {{ font-family: var(--mono); font-size: 96px; line-height: 1; font-weight: 500; color: var(--text); font-variant-numeric: tabular-nums; }}
.purf-specimen .c {{ font-family: var(--mono); font-size: 15px; color: var(--accent); font-variant-numeric: tabular-nums; }}
.purf-specimen .l {{ font-size: 13px; color: var(--muted); }}

/* ---------- Page header ---------- */
.purf-page-head {{ margin: 0 0 24px; }}
.purf-page-head .eyebrow {{ color: var(--muted); font-size: 13px; font-weight: 400; }}
.purf-page-head h1 {{ font-size: 2.25rem; margin: 4px 0 8px; padding: 0; font-weight: 500; color: var(--text); letter-spacing: -0.02em; line-height: 1.15; }}
.purf-page-head p {{ color: var(--text2); margin: 0; max-width: 52rem; line-height: 1.6; }}
.grad-text {{ color: var(--text); }}

/* ---------- Cards and figures ---------- */
.purf-card {{
  position: relative; border-radius: var(--radius); padding: 20px 24px; height: 100%;
  background: var(--surface); border: 1px solid var(--border);
}}
.purf-card h4 {{ margin: 0 0 8px; padding: 0; font-size: 16px; color: var(--text); }}
.purf-card p {{ margin: 0; color: var(--text2); font-size: 14px; line-height: 1.6; }}
.purf-card b {{ color: var(--text); font-weight: 500; }}
.purf-metric {{ position: relative; border-radius: var(--radius); padding: 16px 20px; background: var(--surface); border: 1px solid var(--border); height: 100%; }}
.purf-metric .label {{ color: var(--text2); font-size: 13px; font-weight: 400; }}
.purf-metric .value {{ font-family: var(--mono); font-size: clamp(1.4rem, 2vw, 1.875rem); font-weight: 500; color: var(--text);
  margin-top: 8px; line-height: 1.1; white-space: nowrap; font-variant-numeric: tabular-nums; }}
.purf-metric .sub {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}

.purf-section {{ margin: 48px 0 16px; display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }}
.purf-section h3 {{ margin: 0; padding: 0; font-size: 20px; color: var(--text); }}
.purf-section .line {{ flex: 1; height: 1px; background: var(--border); min-width: 24px; }}
.purf-section .note {{ color: var(--muted); font-size: 13px; }}

.badge {{ display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 4px; font-size: 12px;
  font-weight: 400; border: 1px solid var(--border-strong); color: var(--text2); background: transparent; font-family: var(--mono); }}
.badge.trax {{ color: var(--accent); border-color: rgba(127,167,238,0.35); }}
.badge.good {{ color: #8fd9b0; border-color: rgba(76,195,138,.35); }}
.badge.bad {{ color: #f0a0a2; border-color: rgba(229,72,77,.45); }}

/* ---------- Prediction readout: the digit, flat and large ---------- */
.purf-read {{ display: flex; align-items: center; gap: 16px; padding: 8px 0; flex-wrap: wrap; }}
.purf-read .px {{ display: flex; flex-direction: column; gap: 8px; }}
.purf-read .px img {{ width: 96px; height: 96px; image-rendering: pixelated; background: #000; border-radius: 4px; display: block;
  box-shadow: 0 0 0 1px var(--border); }}
.purf-read .px span {{ font-family: var(--mono); font-size: 12px; color: var(--muted); }}
.purf-read .num {{ display: flex; flex-direction: column; gap: 4px; min-width: 0; flex: 1; }}
.purf-read .digit {{ font-family: var(--mono); font-size: 128px; font-weight: 500; line-height: .9; color: var(--text); font-variant-numeric: tabular-nums; }}
.purf-read .conf {{ font-family: var(--mono); font-size: 24px; font-weight: 500; color: var(--accent); font-variant-numeric: tabular-nums; }}
.purf-read .lbl {{ color: var(--muted); font-size: 13px; }}

.purf-top3 {{ display: flex; flex-direction: column; gap: 12px; }}
.purf-top3 .row {{ display: grid; grid-template-columns: 24px 1fr 64px; gap: 12px; align-items: center; }}
.purf-top3 .d {{ font-family: var(--mono); font-size: 18px; font-weight: 500; text-align: center; color: var(--text); }}
.purf-top3 .bar {{ height: 8px; border-radius: 2px; background: rgba(237,237,237,0.06); overflow: hidden; }}
.purf-top3 .bar span {{ display: block; height: 100%; border-radius: 2px; background: var(--accent); }}
.purf-top3 .row:not(:first-child) .bar span {{ background: {NEUTRAL}; }}
.purf-top3 .p {{ font-family: var(--mono); font-size: 13px; color: var(--text2); text-align: right; font-variant-numeric: tabular-nums; }}

.purf-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
.purf-step {{ border: 1px solid var(--border); border-radius: 12px; padding: 16px; background: var(--surface); }}
.purf-step .n {{ font-family: var(--mono); color: var(--muted); font-size: 12px; }}
.purf-step .t {{ font-weight: 500; margin-top: 8px; color: var(--text); }}
.purf-step .d {{ color: var(--text2); font-size: 13px; margin-top: 4px; line-height: 1.5; }}

.purf-img-tile {{ border: 1px solid var(--border); border-radius: 12px; padding: 8px; background: var(--surface); text-align: center; }}
.purf-img-tile img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #000; image-rendering: pixelated; border-radius: 4px; display: block; }}
.purf-img-tile .cap {{ font-size: 12px; color: var(--text2); margin-top: 8px; line-height: 1.4; font-variant-numeric: tabular-nums; }}
.purf-img-tile .cap b {{ color: var(--text); font-weight: 500; }}
.purf-img-tile .badge {{ font-size: 11px; padding: 0 4px; margin-top: 4px; max-width: 100%; overflow: hidden; }}
.purf-img-tile.bad {{ border-color: rgba(229,72,77,.45); }}

/* Empty 28x28 sheet, used where an icon would otherwise go */
.purf-blank28 {{ width: 112px; height: 112px; border-radius: 4px; background-color: #000;
  background-image: linear-gradient(rgba(237,237,237,0.07) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(237,237,237,0.07) 1px, transparent 1px);
  background-size: 4px 4px; box-shadow: 0 0 0 1px var(--border-strong); margin: 0 auto; }}
.purf-empty {{ display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; gap: 16px; }}
.purf-empty .cap {{ font-family: var(--mono); font-size: 12px; color: var(--muted); }}

.purf-footer {{ margin-top: 64px; padding: 16px 0 8px; border-top: 1px solid var(--border); color: var(--muted); font-size: 13px;
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }}

@media (max-width: 760px) {{
  .purf-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)) !important; }}
  .purf-hero {{ grid-template-columns: 1fr; gap: 24px; }}
  .purf-title {{ font-size: 2.5rem; }}
  .purf-sub {{ font-size: 16px; }}
  .purf-specimen .px {{ width: 128px; height: 128px; }}
  .purf-specimen .d {{ font-size: 72px; }}
  .purf-read .digit {{ font-size: 96px; }}
}}
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


def metric_card(label: str, value: str, sub: str = "", icon: str = "", accent: str | None = None) -> str:
    """A neutral figure card. ``icon`` and ``accent`` are accepted for compatibility and ignored."""
    return (
        f'<div class="purf-metric"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def metric_row(cards: Iterable[str]) -> None:
    import streamlit as st

    cards = list(cards)
    cols = st.columns(len(cards))
    for c, html_card in zip(cols, cards):
        c.markdown(html_card, unsafe_allow_html=True)


def card(title: str, body: str, icon: str = "") -> str:
    """A text card. ``icon`` is accepted for compatibility and ignored."""
    return f'<div class="purf-card"><h4>{esc(title)}</h4><p>{body}</p></div>'


def badge(text: str, kind: str = "") -> str:
    return f'<span class="badge {kind}">{esc(text)}</span>'


def image_tile(img: np.ndarray, caption: str = "", upscale: int = 4, bad: bool = False, rgb: bool = False) -> str:
    uri = rgb_data_uri(img, upscale) if rgb else png_data_uri(img, upscale)
    return (f'<div class="purf-img-tile{" bad" if bad else ""}"><img src="{uri}"/>'
            f'<div class="cap">{caption}</div></div>')


def image_grid(tiles: list[str], columns: int = 6) -> None:
    import streamlit as st

    st.markdown(
        f'<div class="purf-grid" style="--cols:{columns};display:grid;grid-template-columns:repeat({columns},minmax(0,1fr));gap:8px">'
        + "".join(tiles) + "</div>",
        unsafe_allow_html=True,
    )


def prediction_readout(digit: int, confidence: float, label: str = "confidence", img: np.ndarray | None = None) -> str:
    """The predicted digit set large and flat, with its confidence and, optionally, the 28x28 input beside it."""
    px = (f'<div class="px"><img src="{png_data_uri(img, 4)}" alt="28 by 28 model input"/><span>28×28 input</span></div>'
          if img is not None else "")
    return (
        f'<div class="purf-read">{px}<div class="num"><div class="digit">{digit}</div>'
        f'<div class="conf">{confidence * 100:.1f}%</div><div class="lbl">{esc(label)}</div></div></div>'
    )


# Backwards-compatible name
prediction_orb = prediction_readout


def blank28(caption: str = "") -> str:
    """An empty 28x28 sheet: the lab's stand-in for an icon in empty states."""
    cap = f'<div class="cap">{esc(caption)}</div>' if caption else ""
    return f'<div class="purf-blank28"></div>{cap}'


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
        f'<div class="purf-footer"><span>{LAB_NAME} · handwritten digit lab · model {MODEL_NAME}</span>'
        f"<span>MNIST via Kaggle (hojjatk/mnist-dataset) · TensorFlow · scikit-learn · Streamlit</span></div>",
        unsafe_allow_html=True,
    )
