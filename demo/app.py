"""Démo Chat ou Chien : navigation, en-tête commun.

Lancer : streamlit run demo/app.py
"""
import streamlit as st

from ui import apply_styles, hero

st.set_page_config(page_title="Chat ou chien ?", page_icon="🐾", layout="wide", initial_sidebar_state="auto")
apply_styles()

PAGES = [
    st.Page("views/classify.py", title="Tester une photo", icon=":material/photo_camera:", default=True),
    st.Page("views/performance.py", title="Performance", icon=":material/insights:"),
    st.Page("views/method.py", title="Méthode", icon=":material/menu_book:"),
]
SUB = {
    "Tester une photo": "Un réseau de neurones convolutif pré-entraîné sur ImageNet reconnaît les chats et les chiens. "
                        "Envoyez une photo : il répond, montre où il a regardé et signale les images hors sujet.",
    "Performance": "Six modèles comparés sur 2 000 photos jamais vues : du CNN entraîné de zéro au transfert learning "
                   "avec VGG16, ResNet50, InceptionV3 et MobileNetV2.",
    "Méthode": "Données, transfert learning, recherche d'hyperparamètres, fine-tuning et mise en production "
               "(ONNX, FastAPI, Docker, Kubernetes).",
}
page = st.navigation(PAGES, position="sidebar")
hero("Deep learning · Vision par ordinateur", "Chat ou chien ?" if page.title == "Tester une photo" else page.title,
     SUB[page.title])
page.run()
