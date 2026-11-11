"""Méthode : données, transfert learning, recherche d'hyperparamètres, mise en production, limites."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import GITHUB_URL, card, footer, fr, kpi_grid, pct, predictor, reports, section  # noqa: E402

metrics, table, search, hist, quality = reports()
meta = predictor().meta

section("Les données")
kpi_grid([
    ("Photos du jeu Kaggle", fr(quality["files"], 0), "« Dogs vs. Cats » (Asirra), étiquette dans le nom du fichier"),
    ("Doublons écartés", fr(quality["files"] - quality["unique_images"], 0),
     f"{quality['exact_duplicates']} copies exactes, {quality['near_duplicate_groups']} groupes quasi identiques"),
    ("Échantillon utilisé", "12 000", "6 000 chats, 6 000 chiens, tirés au hasard", "hl"),
    ("Découpage", "8 000 / 2 000 / 2 000", "entraînement / validation / test"),
])
st.markdown("<div class='note'>Les doublons sont repérés par empreinte exacte (SHA-256) et perceptuelle (pHash) : une "
            "même photo ne peut pas se retrouver à la fois dans l'entraînement et dans le test, ce qui gonflerait les "
            "scores. Les images ne sont ni copiées ni publiées : le dépôt contient seulement la liste des fichiers et "
            "leurs empreintes.</div>", unsafe_allow_html=True)

section("La démarche")
st.markdown("""<div class='steps'>
<div class='step'><b>1 · Référence</b><p>Un CNN de 4 blocs entraîné de zéro sur 8 000 photos (128 px), avec
augmentation : miroir, rotation, zoom, contraste.</p></div>
<div class='step'><b>2 · Transfert learning</b><p>VGG16, ResNet50, InceptionV3 et MobileNetV2, pré-entraînés sur
ImageNet (1,2 million d'images), gelés : chaque photo devient un vecteur de caractéristiques.</p></div>
<div class='step'><b>3 · Réglages</b><p>Grid search d'une régression logistique et random search (16 essais) d'une tête
de classification par réseau. Choix sur la validation, jamais sur le test.</p></div>
<div class='step'><b>4 · Fine-tuning</b><p>MobileNetV2 : tête entraînée, puis dégel des derniers blocs avec un taux
d'apprentissage 50 fois plus faible pour ajuster les filtres aux chats et aux chiens.</p></div>
</div>""", unsafe_allow_html=True)

section("Mise en production")
l, r = st.columns(2, gap="medium")
k = card(l, "Du modèle à l'API")
k.markdown(f"""<div class='ins'><span class='ins-dot'></span><div>Export <b>ONNX</b> ({meta['model_mb']:.0f} Mo), prétraitement
inclus : l'API n'embarque ni TensorFlow ni Keras, seulement ONNX Runtime (quelques dizaines de Mo).</div></div>
<div class='ins'><span class='ins-dot'></span><div>Réponse en <b>{meta['latency_ms_cpu']:.0f} ms</b> par photo sur processeur ;
écart Keras / ONNX inférieur à {meta['keras_onnx_max_gap']:.0e}.</div></div>
<div class='ins'><span class='ins-dot'></span><div><b>API FastAPI</b> avec clé, contrôle du format par le contenu, limite de
8 Mo, refus des bombes de décompression, aucune photo enregistrée ; modèles vérifiés par empreinte SHA-256.</div></div>
<div class='ins'><span class='ins-dot'></span><div><b>Kubernetes</b> : 2 à 6 pods (autoscaling CPU), sondes de santé,
conteneur non-root en lecture seule, aucun trafic sortant (NetworkPolicy).</div></div>""", unsafe_allow_html=True)
k = card(r, "Limites")
k.markdown(f"""<div class='ins'><span class='ins-dot'></span><div>Deux classes seulement : une photo sans animal reçoit
quand même une réponse. Un second réseau (ImageNet) le signale ; 99 % des vraies photos de chats et chiens passent ce
contrôle.</div></div>
<div class='ins'><span class='ins-dot'></span><div>Réponse « incertain » quand la probabilité « chien » est entre
{pct(meta['unsure_band'][0], 0)} et {pct(meta['unsure_band'][1], 0)}.</div></div>
<div class='ins'><span class='ins-dot'></span><div>Photos d'amateurs, surtout de face ; les races rares, les dessins ou
les photos très sombres sont moins bien reconnus.</div></div>
<div class='ins'><span class='ins-dot'></span><div>Entraîné sur processeur avec un échantillon de 12 000 photos : un GPU
permettrait d'utiliser les 25 000 photos et des images plus grandes.</div></div>
<div class='ins'><span class='ins-dot'></span><div>Code, tests et manifests :
<a href='{GITHUB_URL}' target='_blank' rel='noopener noreferrer'>dépôt GitHub</a>.</div></div>""", unsafe_allow_html=True)
footer()
