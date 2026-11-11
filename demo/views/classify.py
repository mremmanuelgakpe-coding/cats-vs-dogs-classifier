"""Tester une photo : prédiction, garde-fou et carte d'occlusion."""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import card, footer, fr, kpi_grid, pct, predictor, section  # noqa: E402

from catdog.inference import ImageError, decode  # noqa: E402

P = predictor()
lo, hi = P.band

section("Votre photo", "Envoyez une photo de chat ou de chien (JPEG, PNG ou WEBP, 8 Mo maximum), ou prenez-en une avec "
                       "la caméra. Elle est analysée en mémoire puis oubliée.")
left, right = st.columns([1, 1.15], gap="medium")
src = card(left, "Image à analyser")
mode = src.segmented_control("Source", ["Fichier", "Caméra"], default="Fichier", label_visibility="collapsed", key="src")
upload = (src.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
          if mode != "Caméra" else src.camera_input("Photo", label_visibility="collapsed"))

if upload is None:
    res = card(right, "Résultat")
    res.markdown("<div class='tip'><b>En attente d'une photo.</b> Le modèle répond « Chat », « Chien » ou « Incertain », "
                 "vérifie que l'image ressemble bien à un animal de compagnie et montre les zones qui ont compté.</div>",
                 unsafe_allow_html=True)
    m = P.meta
    kpi_grid([("Exactitude sur le test", pct(m["test"]["accuracy"]), "2 000 photos jamais vues", "hl"),
              ("Modèle", m["label"], f"{m['model_mb']:.0f} Mo, ONNX"),
              ("Temps de réponse", f"{m['latency_ms_cpu']:.0f} ms", "par photo, sur processeur"),
              ("Zone « incertain »", f"{pct(lo, 0)} – {pct(hi, 0)}", "probabilité « chien » entre ces bornes")])
    footer()
    st.stop()

try:
    im = decode(upload.getvalue())
except ImageError as exc:
    st.error(str(exc))
    st.stop()

src.image(im, use_container_width=True)
r = P.predict(im)
name = {"cat": "Chat", "dog": "Chien", "unsure": "Incertain"}[r.label]


def prob(v):
    return "plus de 99,9\u00a0%" if v >= 0.9995 else "moins de 0,1\u00a0%" if v < 0.0005 else pct(v)


sub = {"cat": f"Probabilité « chat » : {prob(1 - r.probability_dog)}",
       "dog": f"Probabilité « chien » : {prob(r.probability_dog)}",
       "unsure": f"Le modèle hésite : {pct(r.probability_dog)} « chien », {pct(1 - r.probability_dog)} « chat »."}[r.label]
res = card(right, "Résultat")
res.markdown(f"<div class='result {r.label}'><div class='r-label'>Le modèle voit</div><div class='r-value'>{name}</div>"
             f"<div class='r-sub'>{sub}</div></div>"
             f"<div class='gauge'><div class='g-track'><div class='g-mark' style='left:calc({min(max(r.probability_dog, 0.005), 0.995) * 100:.1f}% - 3px)'>"
             f"</div></div><div class='g-lab'><span>Chat</span><span>Incertain</span><span>Chien</span></div></div>",
             unsafe_allow_html=True)
pet_txt = "oui" if r.looks_like_pet else "non"
res.markdown(f"""<div style='height:.9rem'></div>
<div class='ins'><span class='ins-dot'></span><div>Confiance dans la réponse : <b>{prob(r.confidence)}</b>
(réponse « incertain » entre {pct(lo, 0)} et {pct(hi, 0)} de probabilité « chien »).</div></div>
<div class='ins'><span class='ins-dot'></span><div>Ressemble à un chat ou à un chien : <b>{pet_txt}</b>
(score {fr(r.pet_score, 2)}, seuil {fr(P.gate_threshold, 2)}).</div></div>
<div class='ins'><span class='ins-dot'></span><div>Modèle : <b>{P.meta['label']}</b> pré-entraîné sur ImageNet, tête
réglée par recherche aléatoire, {pct(P.meta['test']['accuracy'])} d'exactitude sur 2 000 photos de test.</div></div>""",
             unsafe_allow_html=True)
if not r.looks_like_pet:
    res.markdown("<div class='warn'><b>Cette image ne ressemble ni à un chat ni à un chien.</b> Le modèle ne connaît que "
                 "ces deux classes : il répond quand même, mais sa réponse n'a pas de sens ici. Ce contrôle vient d'un "
                 "second réseau (MobileNetV2, 1 000 classes ImageNet).</div>", unsafe_allow_html=True)

with st.spinner("Calcul des zones importantes…"):
    occ = P.occlusion(im)
section("Ce qui a compté pour la décision", "Carte d'occlusion : on masque tour à tour chaque case d'une grille 8 × 8 et "
                                            "on mesure de combien la confiance du modèle baisse.")
l2, r2 = st.columns([1, 1.15], gap="medium")
k = card(l2, "Zones importantes", "Plus c'est orange, plus la zone a pesé dans la réponse.")
w, h = im.size
heat = occ / occ.max() if occ.max() > 0 else occ
heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
alpha = np.asarray(heat_img, dtype=np.float32)[..., None] / 255 * 0.75
overlay = np.asarray(im, dtype=np.float32) * (1 - alpha) + np.array([244, 114, 30], np.float32) * alpha
k.image(Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)), use_container_width=True)
k = card(r2, "Lecture")
top = np.argsort(occ.ravel())[::-1][:3]
rows = ["haut", "milieu haut", "milieu", "milieu bas", "bas"]
cols = ["gauche", "centre gauche", "centre", "centre droit", "droite"]
where = [f"{rows[(i // 8) * 5 // 8]} {cols[(i % 8) * 5 // 8]}" for i in top]
k.markdown(f"""<div class='ins'><span class='ins-dot'></span><div>Zones les plus utiles : <b>{where[0]}</b>, puis
<b>{where[1]}</b> et <b>{where[2]}</b>.</div></div>
<div class='ins'><span class='ins-dot'></span><div>Masquer la zone la plus utile fait baisser la probabilité de la
réponse de <b>{fr(100 * float(occ.max()), 0)} points</b>.</div></div>
<div class='ins'><span class='ins-dot'></span><div>Sur une bonne prédiction, la carte se concentre en général sur la tête
(yeux, museau, oreilles). Si elle s'allume sur le décor, le modèle s'appuie sur le contexte : prudence.</div></div>""",
           unsafe_allow_html=True)
k.markdown("<div class='privacy'>Aucune photo n'est enregistrée : l'image reste en mémoire le temps du calcul.</div>",
           unsafe_allow_html=True)
footer()
