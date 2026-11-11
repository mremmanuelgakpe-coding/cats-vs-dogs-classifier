"""Socle de la démo : palette du portfolio, styles, formats français, chargement sécurisé du modèle ONNX."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

GITHUB_URL = "https://github.com/mremmanuelgakpe-coding/cats-vs-dogs-classifier"
LINKEDIN_URL = "https://www.linkedin.com/in/egakpe"
CONTACT_MAIL = "mailto:mremmanuelgakpe@icloud.com?subject=Portfolio%20Data%20%26%20IA%20-%20Chat%20ou%20chien"
LINKS = (f'<a href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer">LinkedIn</a> · '
         f'<a href="{CONTACT_MAIL}">Me contacter</a> · '
         f'<a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer">Code</a>')

# Palette du projet 7 : bordeaux, vert, crème, cartes blanches
CREAM, GREEN, BORDO, WHITE = "#F5DABF", "#0F3D3A", "#6C151E", "#FFFFFF"
INK, MUTED, GRID = "#1B2524", "#5E6866", "#EFE2D4"
DATA_GREEN, DATA_GREEN_LIGHT, NEUTRAL = "#0F3D3A", "#7FA8A2", "#D9C3AA"
BORDO_RAMP = ["#FBF3F3", "#F3DCDD", "#DFA3AA", "#C0606C", "#6C151E", "#4A0E14"]
CHART_FONT = "Montserrat, system-ui, sans-serif"


def fr(x, nd=1):
    return f"{x:,.{nd}f}".replace(",", "\u00a0").replace(".", ",")


def pct(x, nd=1):
    return f"{fr(100 * x, nd)}\u00a0%"


@st.cache_resource
def predictor():
    from catdog.inference import Predictor
    return Predictor(ROOT / "models")       # refuse un modèle dont l'empreinte SHA-256 ne correspond pas


@st.cache_data
def reports():
    r = ROOT / "reports"
    return (json.loads((r / "metrics.json").read_text()), pd.read_csv(r / "model_comparison.csv"),
            json.loads((r / "transfer_search.json").read_text()),
            {n: json.loads((r / f"history_{n}.json").read_text()) for n in ("cnn_scratch", "mobilenet_v2_ft")
             if (r / f"history_{n}.json").exists()},
            json.loads((r / "data_quality.json").read_text()))


def hero(eyebrow, title, subtitle):
    st.markdown(f"""<div class="herob"><div class="hb-top"><span class="eyebrow">{eyebrow}</span>
<span class="hb-meta">Projet portfolio de <b>Emmanuel</b> · {LINKS} · {date.today().strftime('%d/%m/%Y')}</span></div>
<h1>{title}</h1><p>{subtitle}</p></div>""", unsafe_allow_html=True)


def section(title, caption=None):
    cap = f"<div class='sec-cap'>{caption}</div>" if caption else ""
    st.markdown(f"<div class='sec'><span></span>{title}</div>{cap}", unsafe_allow_html=True)


def kpi_grid(items, cols=4):
    """Cartes de même taille. items : (libellé, valeur, sous-texte, ton) ; ton = '', 'orange', 'blue', 'hl'."""
    cards = "".join(f"<div class='kcard {rest[0] if rest else ''}'><div class='k-label'>{label}</div>"
                    f"<div class='k-value num'>{value}</div><div class='k-sub'>{sub}</div></div>"
                    for label, value, sub, *rest in items)
    st.markdown(f"<div class='kgrid c{cols}'>{cards}</div>", unsafe_allow_html=True)


def card(parent=None, title=None, caption=None, key=None):
    st.session_state["_card_seq"] = st.session_state.get("_card_seq", 0) + 1
    c = (parent or st).container(key=key or f"card-{st.session_state['_card_seq']}",
                                 height="stretch" if parent else "content")
    if title:
        cap = f"<div class='c-cap'>{caption}</div>" if caption else ""
        c.markdown(f"<div class='c-title'>{title}</div>{cap}", unsafe_allow_html=True)
    return c


def chart_config(chart, height):
    return (chart.properties(height=height).configure_view(strokeWidth=0)
            .configure(background="transparent", font=CHART_FONT)
            .configure_axis(labelFontSize=14, titleFontSize=14, titleFontWeight=500, labelColor=MUTED,
                            titleColor=MUTED, gridColor=GRID, domainColor="#E2CDB6", tickColor="#E2CDB6")
            .configure_legend(labelFontSize=14, titleFontSize=14, labelColor=INK, titleColor=MUTED))


def footer():
    st.markdown("<div class='foot'>Démonstration pédagogique. Les photos envoyées sont analysées en mémoire puis "
                "oubliées : rien n'est enregistré. Une question, une mission ou un poste à pourvoir ? Échangeons sur "
                f"<a href='{LINKEDIN_URL}' target='_blank' rel='noopener noreferrer'>LinkedIn</a> ou "
                f"<a href='{CONTACT_MAIL}'>par e-mail</a>.</div>", unsafe_allow_html=True)


CSS = f"""
@font-face {{ font-family:'Montserrat'; src:url('app/static/fonts/montserrat-latin-wght-normal.woff2') format('woff2');
             font-weight:100 900; font-style:normal; font-display:swap; }}
@font-face {{ font-family:'Montserrat'; src:url('app/static/fonts/montserrat-latin-wght-italic.woff2') format('woff2');
             font-weight:100 900; font-style:italic; font-display:swap; }}
:root {{ --page:{CREAM}; --panel:#FBF1E6; --card:{WHITE}; --ink:{INK}; --muted:{MUTED}; --orange:{BORDO};
        --blue:{GREEN}; --blue-deep:#0C312F; --tint:#F3DCDD; --line:#F1E3D3; }}
.stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button {{ font-family:'Montserrat', system-ui, sans-serif; }}
[data-testid="stTooltipIcon"] button {{ background:transparent !important; color:var(--muted) !important; }}
.stApp {{ background:var(--page); color:var(--ink); }}
header[data-testid="stHeader"] {{ background:var(--page); }}
.block-container {{ max-width:1280px; padding:1.4rem 1.6rem 3rem; background:var(--panel); border-radius:30px;
                    margin-top:1.2rem; margin-bottom:1.5rem; }}
h1, h2, h3 {{ font-family:'Montserrat', sans-serif !important; color:var(--ink); letter-spacing:-0.02em; }}
.num {{ font-variant-numeric:tabular-nums; }}

/* navigation entre les vues (barre du haut) */
header[data-testid="stHeader"] a[data-testid="stTopNavLink"] {{ border-radius:999px; padding:.35rem 1rem; }}
header[data-testid="stHeader"] a[data-testid="stTopNavLink"] span {{ font-size:.95rem; font-weight:600; color:var(--blue); }}
header[data-testid="stHeader"] a[data-testid="stTopNavLink"][aria-current="page"] {{ background:var(--card); }}
header[data-testid="stHeader"] a[data-testid="stTopNavLink"][aria-current="page"] span {{ color:var(--orange); }}

.topbar {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; }}
.brand {{ display:flex; align-items:center; gap:.5rem; font-weight:700; font-size:1.15rem; letter-spacing:-0.03em; color:var(--blue); }}
.brand-mark {{ width:26px; height:26px; border-radius:8px; background:var(--blue); position:relative; }}
.brand-mark::after {{ content:""; position:absolute; right:-4px; bottom:-4px; width:12px; height:12px; border-radius:50%;
                      background:var(--orange); border:3px solid var(--panel); }}
.author {{ color:var(--muted); font-size:.875rem; }}
.author a {{ color:var(--ink); font-weight:600; text-decoration:none; border-bottom:1.5px solid var(--orange); }}
.title-row {{ display:flex; align-items:center; gap:.75rem; margin:1.1rem 0 .3rem; flex-wrap:wrap; }}
.title-row h1 {{ font-size:1.9rem; font-weight:600; margin:0; padding:0; }}
.chip {{ background:var(--card); border-radius:999px; padding:.25rem .7rem; font-size:.875rem; color:var(--muted); }}
.lead {{ color:var(--muted); font-size:.92rem; max-width:75ch; margin-bottom:.6rem; }}

.stTabs [role="tablist"] {{ gap:.25rem; background:var(--card); border-radius:999px; padding:.3rem; width:fit-content;
                            max-width:100%; overflow-x:auto; border:none !important; box-shadow:none !important; }}
.stTabs [role="tab"] {{ border-radius:999px; padding:.4rem 1rem; color:var(--muted); border:none !important; }}
.stTabs [role="tab"] p {{ font-size:.875rem; font-weight:500; }}
.stTabs [role="tab"][data-selected="true"] {{ background:var(--panel); }}
.stTabs [role="tab"][data-selected="true"] p {{ color:var(--orange); font-weight:600; }}
.stTabs [role="tab"]:focus-visible {{ outline:3px solid var(--orange); outline-offset:1px; }}
.stTabs .react-aria-SelectionIndicator {{ display:none !important; }}

.card {{ background:var(--card); border-radius:22px; padding:1.05rem 1.2rem; height:100%; margin-bottom:12px; }}
.card .label, .hero .label, .bluecard .label {{ font-size:.92rem; font-weight:500; }}
.card .label {{ color:var(--ink); }}
.card .value, .bluecard .value {{ font-size:1.8rem; font-weight:600; letter-spacing:-0.03em; margin-top:.25rem; }}
.card .sub {{ font-size:.875rem; color:var(--muted); margin-top:.2rem; }}
.hero {{ background:radial-gradient(120% 120% at 100% 0%, #A3323E 0%, {BORDO} 45%, #4A0E14 100%); color:#fff;
         border-radius:22px; padding:1.05rem 1.2rem; height:100%; margin-bottom:12px; position:relative; overflow:hidden; }}
.hero .label {{ color:#FBEFEF; }}
.hero .value {{ font-size:2.4rem; font-weight:600; letter-spacing:-0.04em; line-height:1.1; margin-top:.2rem; }}
.hero .sub {{ color:#FBEFEF; font-size:.875rem; margin-top:.2rem; }}
.bluecard {{ background:var(--blue); color:#fff; border-radius:22px; padding:1.05rem 1.2rem; height:100%; margin-bottom:12px; }}
.bluecard .label, .bluecard .sub {{ color:#E2EEEC; }}
.bluecard .sub {{ font-size:.875rem; margin-top:.2rem; }}
.icon {{ width:26px; height:26px; border-radius:7px; background:var(--blue); color:#fff; display:inline-flex;
         align-items:center; justify-content:center; font-size:.875rem; font-weight:700; margin-bottom:.7rem; }}
.hero .icon {{ background:#fff; color:var(--orange); }}
.badge {{ position:absolute; top:1rem; right:1rem; background:rgba(255,255,255,.22); border-radius:999px; padding:.15rem .6rem;
          font-size:.875rem; }}
.badge.light {{ background:var(--panel); color:var(--muted); }}
.decision {{ display:inline-block; border-radius:999px; padding:.2rem .8rem; font-weight:600; font-size:1.1rem; margin-top:.3rem; }}
.decision.ok {{ background:#E2EEEC; color:var(--blue); }}
.decision.ko {{ background:var(--tint); color:#6C151E; }}

.scale {{ display:flex; gap:6px; margin-top:.8rem; }}
.seg {{ flex:1; height:46px; border-radius:12px; position:relative; }}
.seg.on {{ outline:3px solid var(--blue); outline-offset:2px; }}
.seg span {{ position:absolute; bottom:-1.6rem; left:0; right:0; text-align:center; font-size:.875rem; color:var(--muted); }}
.scale-top {{ display:flex; justify-content:space-between; font-size:.875rem; color:var(--muted); }}
.scale-top b {{ color:var(--ink); font-size:1.2rem; font-weight:600; }}

.dark {{ background:linear-gradient(160deg, var(--blue) 0%, var(--blue-deep) 100%); color:#FBF5EE; border-radius:24px;
         padding:1.3rem 1.3rem 1rem; }}
.dark h3 {{ color:#fff; font-size:1.15rem; font-weight:500; margin:0 0 .2rem; }}
.dark .hint {{ color:#CFE2DF; font-size:.875rem; margin-bottom:.8rem; }}
.dark .count {{ float:right; background:var(--orange); color:#fff; border-radius:999px; padding:.05rem .6rem; font-size:.875rem; font-weight:700; }}
.reason {{ display:flex; justify-content:space-between; align-items:center; gap:.8rem; padding:.55rem 0;
           border-top:1px solid rgba(255,255,255,.16); }}
.reason .t {{ font-size:.9rem; color:#fff; }}
.reason .v {{ font-size:.875rem; color:#CFE2DF; }}
.dot {{ width:24px; height:24px; flex:0 0 24px; border-radius:50%; display:flex; align-items:center; justify-content:center;
        font-size:.875rem; font-weight:700; }}
.dot.up {{ background:var(--orange); color:#fff; }}
.dot.down {{ background:#fff; color:var(--blue); }}
.group {{ font-size:.875rem; color:#CFE2DF; margin:.7rem 0 .1rem; }}

[class*="st-key-box"], .st-key-form, .st-key-chart {{ background:var(--card); border-radius:22px; padding:1.1rem 1.2rem 1.2rem; }}
[class*="st-key-box"] h3, .st-key-form h3, .st-key-chart h3 {{ font-size:1.1rem; font-weight:600; margin:0; }}
.st-key-chart, [class*="st-key-box"] {{ margin-top:12px; }}
div[data-baseweb="select"] > div, .stNumberInput input {{ border-radius:12px !important; background:#FDF8F2 !important; }}
.stButton button, [data-testid="stPageLink"] a {{ border-radius:999px; border:none; background:var(--panel); color:var(--ink);
                                                  font-weight:600; font-size:.875rem; }}
.stButton button:hover {{ background:var(--tint); color:#6C151E; }}
.stButton button:focus-visible, [data-testid="stPageLink"] a:focus-visible {{ outline:3px solid var(--orange); outline-offset:2px; }}
.st-key-goto a, .st-key-goto2 a {{ background:var(--blue) !important; padding:.45rem 1.1rem !important; width:fit-content; max-width:100%; height:auto; }}
.st-key-goto a p, .st-key-goto2 a p {{ color:#fff !important; font-weight:600; white-space:normal !important; overflow:visible !important; text-overflow:clip !important; }}

[data-testid="stButtonGroup"] button[aria-checked="false"] {{ background:#FFFFFF; color:var(--ink); border-color:#EBD9C6; }}
[data-testid="stButtonGroup"] button[aria-checked="false"]:hover {{ background:var(--panel); color:var(--ink); }}
.slegend {{ display:flex; gap:1.2rem; font-size:.875rem; color:var(--muted); margin:.4rem 0 .6rem; flex-wrap:wrap; }}
.slegend i {{ display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:.4rem; vertical-align:-1px; }}
.srow {{ display:grid; grid-template-columns:minmax(170px, 1.1fr) 2fr 70px; gap:.9rem; align-items:center; padding:.45rem 0;
         border-top:1px solid var(--line); }}
.slab {{ font-size:.9rem; color:var(--ink); }}
.slab span {{ display:block; font-size:.875rem; color:var(--muted); }}
.strack {{ position:relative; height:14px; background:#F8ECDF; border-radius:4px; }}
.strack.center {{ background:transparent; }}
.smid {{ position:absolute; left:50%; top:-6px; bottom:-6px; width:1.5px; background:#E2CDB6; }}
.sbar {{ position:absolute; top:0; height:14px; border-radius:4px; min-width:2px; }}
.sval {{ font-size:.875rem; font-weight:600; text-align:right; }}

.grid {{ display:grid; grid-template-columns:repeat(10, 1fr); gap:6px; margin-top:.8rem; }}
.dcol {{ display:flex; flex-direction:column; align-items:center; gap:4px; }}
.dcol .cell {{ width:100%; }}
.cell {{ height:40px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:.875rem; font-weight:600; }}
.rowlab {{ font-size:.875rem; color:var(--muted); text-align:center; }}
.note {{ background:#FBF3F3; border-radius:14px; padding:.6rem .9rem; font-size:.875rem; color:var(--muted); margin:.4rem 0; }}
.foot {{ color:var(--muted); font-size:.875rem; margin-top:1.4rem; }}
@media (max-width: 640px) {{
  .block-container {{ padding:1rem .9rem 2rem; border-radius:22px; margin-top:.6rem; }}
  .hero .value {{ font-size:2rem; }} .card .value, .bluecard .value {{ font-size:1.5rem; }}
  .grid {{ grid-template-columns:repeat(5, 1fr); gap:8px; }}
  .srow {{ grid-template-columns:1fr 64px; gap:.3rem .6rem; }}
  .srow .strack {{ grid-column:1 / -1; grid-row:2; }}
}}
"""




CSS_DASH = """
[data-testid="stSidebar"] { background:linear-gradient(180deg, #0F3D3A 0%, #0C312F 60%, #082624 100%) !important; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top:.4rem; }
[data-testid="stSidebarNavLink"] { border-radius:999px !important; margin:2px 4px; padding:.35rem .8rem !important; }
[data-testid="stSidebarNavLink"] span { color:#E8F2F0 !important; font-size:.95rem !important; font-weight:500; }
[data-testid="stSidebarNavLink"]:hover { background:rgba(255,255,255,.10) !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background:#6C151E !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] span { color:#FFFFFF !important; font-weight:700; }
[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] { color:#FFFFFF !important; }
[data-testid="stNavSectionHeader"], [data-testid="stNavSectionHeader"] span { color:#F5DABF !important; font-size:.875rem !important;
  font-weight:700 !important; letter-spacing:.08em; text-transform:uppercase; }
[data-testid="stSidebarNavSeparator"] { border-color:rgba(255,255,255,.18) !important; }
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { display:none; }
.sbbrand { display:flex; align-items:center; gap:.6rem; color:#fff; font-weight:700; font-size:1.15rem; margin:.4rem .3rem 1rem; }
.sbbrand i { width:28px; height:28px; border-radius:9px; background:#fff; position:relative; display:inline-block; }
.sbbrand i::after { content:""; position:absolute; right:-4px; bottom:-4px; width:12px; height:12px; border-radius:50%;
  background:#6C151E; border:3px solid #0F3D3A; }
.sbbrand span { display:block; font-size:.875rem; font-weight:500; color:#CFE2DF; }
.st-key-sidebar-filters { background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14); border-radius:20px;
  padding:14px 12px 10px; margin-top:.6rem; }
.sbtitle { color:#fff; font-weight:700; font-size:1rem; }
.st-key-sidebar-filters label p { color:#E8F2F0 !important; font-size:.875rem !important; font-weight:600; }
.st-key-sidebar-filters [data-testid="stMultiSelect"] div { background-color:#FFFFFF !important; }
.st-key-sidebar-filters [data-testid="stWidgetLabel"], .st-key-sidebar-filters [data-testid="stWidgetLabel"] div { background-color:transparent !important; }
.st-key-sidebar-filters [data-testid="stMultiSelect"] [data-baseweb="tag"] { background-color:#6C151E !important; }
.st-key-sidebar-filters [data-testid="stMultiSelect"] [data-baseweb="tag"] div { background-color:transparent !important; }
.st-key-sidebar-filters .stButton button { background:#FFFFFF; color:#0C312F; }
[data-baseweb="tag"] { background:#6C151E !important; border-radius:999px !important; }
[data-baseweb="tag"] span { color:#fff !important; font-size:.875rem !important; }
.sbnote { color:#CFE2DF; font-size:.875rem; margin:.8rem .2rem 0; line-height:1.45; }

.herob { position:relative; overflow:hidden; border-radius:26px; padding:1.4rem 1.8rem 1.4rem; margin-bottom:1rem; color:#fff;
  background:radial-gradient(70% 140% at 100% 0%, rgba(163,50,62,.85) 0%, rgba(163,50,62,0) 60%),
             radial-gradient(50% 120% at 80% 130%, rgba(163,50,62,.45) 0%, rgba(163,50,62,0) 60%),
             linear-gradient(135deg, #0F3D3A 0%, #082624 100%);
  box-shadow:0 16px 36px rgba(15,61,58,.22); }
.hb-top { display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; align-items:center; }
.eyebrow { font-size:.875rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:#F5DABF; }
.hb-meta { font-size:.875rem; color:#E2EEEC; }
.foot a { color:#1B2524; font-weight:600; text-decoration:none; border-bottom:1.5px solid #6C151E; }
.hb-meta a { color:#fff; font-weight:600; text-decoration:none; border-bottom:1.5px solid #6C151E; }
.herob h1 { color:#fff !important; font-size:2rem; font-weight:700; margin:.5rem 0 .2rem; padding:0; letter-spacing:-.02em; }
.herob p { color:#E2EEEC; font-size:.95rem; margin:0; max-width:80ch; }

.sec { display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.1rem; color:#1B2524; margin:1.1rem 0 .55rem; }
.sec span { width:6px; height:20px; border-radius:6px; background:#6C151E; display:inline-block; }
.sec-cap { color:#5E6866; font-size:.875rem; margin:-.3rem 0 .6rem; }

.kgrid { display:grid; gap:14px; margin-bottom:1rem; }
.kgrid.c4 { grid-template-columns:repeat(4, minmax(0, 1fr)); }
.kgrid.c3 { grid-template-columns:repeat(3, minmax(0, 1fr)); }
.kcard { background:#FFFFFF; border-radius:20px; padding:1rem 1.15rem; display:flex; flex-direction:column; min-height:138px;
  border-top:4px solid #0F3D3A; box-shadow:0 1px 2px rgba(27,37,36,.04), 0 10px 26px rgba(27,37,36,.06);
  transition:transform .15s ease, box-shadow .15s ease; }
.kcard:hover { transform:translateY(-2px); box-shadow:0 1px 2px rgba(27,37,36,.05), 0 16px 30px rgba(27,37,36,.10); }
.kcard.orange { border-top-color:#6C151E; }
.kcard.hl { border-top-color:transparent; color:#fff;
  background:radial-gradient(120% 120% at 100% 0%, #A3323E 0%, #6C151E 45%, #4A0E14 100%); }
.kcard.blue { border-top-color:transparent; color:#fff; background:linear-gradient(150deg, #0F3D3A 0%, #0C312F 100%); }
.k-label { font-size:.875rem; font-weight:600; color:#5E6866; }
.k-value { font-size:1.75rem; font-weight:700; letter-spacing:-.03em; color:#1B2524; margin:.35rem 0 .3rem; line-height:1.15; }
.k-sub { font-size:.875rem; color:#5E6866; margin-top:auto; line-height:1.4; }
.kcard.hl .k-label, .kcard.hl .k-sub, .kcard.blue .k-label, .kcard.blue .k-sub { color:rgba(255,255,255,.88); }
.kcard.hl .k-value, .kcard.blue .k-value { color:#fff; }

[class*="st-key-card-"] { background:#FFFFFF; border-radius:22px; padding:1.1rem 1.25rem 1.1rem !important;
  box-shadow:0 1px 2px rgba(27,37,36,.04), 0 10px 26px rgba(27,37,36,.06); }
.c-title { font-weight:700; font-size:1.02rem; color:#1B2524; }
.c-cap { color:#5E6866; font-size:.875rem; margin-top:.15rem; line-height:1.45; }
.dleg { margin-top:.2rem; }
.dl { display:grid; grid-template-columns:14px 1fr auto 52px; gap:.6rem; align-items:center; padding:.42rem 0;
  border-top:1px solid #F1E3D3; font-size:.875rem; }
.dl.off { opacity:.4; }
.dl i { width:12px; height:12px; border-radius:4px; display:inline-block; }
.dl-n { color:#1B2524; } .dl-v { color:#5E6866; text-align:right; } .dl-p { font-weight:700; text-align:right; color:#1B2524; }
.ins { display:flex; gap:.75rem; align-items:flex-start; padding:.6rem 0; border-top:1px solid #F1E3D3; font-size:.92rem; color:#1B2524; }
.ins:first-of-type { border-top:none; }
.ins-dot { width:10px; height:10px; border-radius:50%; background:#6C151E; margin-top:.4rem; flex:0 0 10px; }
[data-testid="stElementToolbar"] { display:none !important; }
.tip { background:#FFFFFF; border-radius:14px; padding:.6rem .9rem; font-size:.875rem; color:#5E6866; margin:.2rem 0 .4rem;
  border-left:4px solid #6C151E; }
.tip b { color:#1B2524; }
.st-key-chips { align-items:center; flex-wrap:wrap; margin:.2rem 0 .4rem; }
.chips-lab { font-size:.875rem; font-weight:700; color:#1B2524; margin-right:.2rem; }
.st-key-chips .stButton button { background:#FFFFFF; border:1.5px solid #6C151E; color:#6C151E; padding:.2rem .8rem;
  min-height:0; font-size:.875rem; }
.st-key-chips .stButton button:hover { background:#F3DCDD; }
.st-key-chips [class*="st-key-chip_clear"] button { background:#0F3D3A; border-color:#0F3D3A; color:#FFFFFF; }
.srow.wide { grid-template-columns:minmax(0, 1.5fr) minmax(80px, 1fr) 56px; }
.ogrid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:14px; }
.ocard { background:#FFFFFF; border-radius:20px; padding:1rem 1.15rem; box-shadow:0 1px 2px rgba(27,37,36,.04), 0 10px 26px rgba(27,37,36,.06);
  display:flex; flex-direction:column; gap:.45rem; min-height:170px; }
.ocard .o-name { font-weight:700; font-size:.95rem; color:#1B2524; }
.ocard .o-val { font-size:1.6rem; font-weight:700; color:#1B2524; letter-spacing:-.02em; }
.ocard .o-val small { font-size:.875rem; font-weight:500; color:#5E6866; margin-left:.3rem; }
.otrack { height:10px; border-radius:6px; background:#F8ECDF; position:relative; overflow:hidden; }
.otrack div { position:absolute; left:0; top:0; bottom:0; border-radius:6px; background:#0F3D3A; }
.o-cmp { display:flex; justify-content:space-between; font-size:.875rem; color:#5E6866; margin-top:auto; }
.o-cmp b { color:#1B2524; }
.st-key-tomodel a { background:#0F3D3A !important; padding:.45rem 1.1rem !important; width:fit-content; max-width:100%; height:auto; }
.st-key-tomodel a p { color:#fff !important; font-weight:600; white-space:normal !important; }
@media (max-width: 1100px) {
  .kgrid.c4 { grid-template-columns:repeat(2, minmax(0, 1fr)); }
  .ogrid { grid-template-columns:repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
  .kgrid.c4, .kgrid.c3, .ogrid { grid-template-columns:repeat(2, minmax(0, 1fr)); gap:10px; }
  [data-testid="stElementToolbar"] { display:none !important; }
.tip { background:#FFFFFF; border-radius:14px; padding:.6rem .9rem; font-size:.875rem; color:#5E6866; margin:.2rem 0 .4rem;
  border-left:4px solid #6C151E; }
.tip b { color:#1B2524; }
.st-key-chips { align-items:center; flex-wrap:wrap; margin:.2rem 0 .4rem; }
.chips-lab { font-size:.875rem; font-weight:700; color:#1B2524; margin-right:.2rem; }
.st-key-chips .stButton button { background:#FFFFFF; border:1.5px solid #6C151E; color:#6C151E; padding:.2rem .8rem;
  min-height:0; font-size:.875rem; }
.st-key-chips .stButton button:hover { background:#F3DCDD; }
.st-key-chips [class*="st-key-chip_clear"] button { background:#0F3D3A; border-color:#0F3D3A; color:#FFFFFF; }
.srow.wide { grid-template-columns:1fr 64px; }
  .kcard { min-height:128px; padding:.85rem .9rem; } .k-value { font-size:1.35rem; }
  .herob { padding:1.1rem 1.1rem; } .herob h1 { font-size:1.5rem; }
}
"""

CSS_EXTRA = """
.result { border-radius:24px; padding:1.3rem 1.4rem; color:#fff; position:relative; overflow:hidden; }
.result.cat { background:radial-gradient(120% 120% at 100% 0%, #A3323E 0%, #6C151E 45%, #4A0E14 100%); }
.result.dog { background:linear-gradient(150deg, #0F3D3A 0%, #082624 100%); }
.result.unsure { background:linear-gradient(150deg, #8C8178 0%, #5E5650 100%); }
.result .r-label { font-size:.875rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; opacity:.9; }
.result .r-value { font-size:2.6rem; font-weight:700; letter-spacing:-.03em; line-height:1.1; margin:.3rem 0; }
.result .r-sub { font-size:.95rem; opacity:.92; }
.gauge { margin-top:1rem; }
.gauge .g-track { height:14px; border-radius:8px; background:linear-gradient(90deg, #6C151E 0%, #6C151E 20%, #F5DABF 20%, #F5DABF 80%, #0F3D3A 80%); position:relative; }
.gauge .g-mark { position:absolute; top:-6px; width:6px; height:26px; border-radius:4px; background:#1B2524; box-shadow:0 0 0 3px #FFFFFF; }
.gauge .g-lab { display:flex; justify-content:space-between; font-size:.875rem; color:#5E6866; margin-top:.5rem; }
.warn { background:#FBEFEF; border-left:4px solid #6C151E; border-radius:14px; padding:.75rem 1rem; font-size:.92rem; color:#1B2524; margin-top:.8rem; }
.privacy { font-size:.875rem; color:#5E6866; margin-top:.6rem; }
.steps { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:14px; }
.step { background:#FFFFFF; border-radius:20px; padding:1rem 1.15rem; box-shadow:0 1px 2px rgba(27,37,36,.04), 0 10px 26px rgba(27,37,36,.06); }
.step b { display:block; color:#6C151E; font-size:.875rem; letter-spacing:.08em; text-transform:uppercase; margin-bottom:.3rem; }
.step p { font-size:.92rem; color:#1B2524; margin:0; line-height:1.5; }
@media (max-width: 900px) { .steps { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
@media (max-width: 640px) { .steps { grid-template-columns:1fr; } .result .r-value { font-size:2rem; } }
"""


def apply_styles():
    st.session_state["_card_seq"] = 0
    st.markdown(f"<style>{CSS}{CSS_DASH}{CSS_EXTRA}</style>", unsafe_allow_html=True)
