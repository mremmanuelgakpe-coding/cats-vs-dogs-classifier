"""Performance : comparaison des 6 modèles, matrice de confusion, apprentissage et recherche d'hyperparamètres."""
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import (DATA_BLUE, INK, ORANGE, ORANGE_RAMP, card, chart_config, footer, fr, kpi_grid, pct,  # noqa: E402
                reports, section)

metrics, table, search, hist, _ = reports()
dep = table[table["deployed"]].iloc[0]
cm = metrics["deployed_confusion"]
n = metrics["test_size"]

section("Modèle déployé sur le test", f"{fr(n, 0)} photos jamais utilisées pour l'entraînement ni pour les réglages.")
kpi_grid([
    ("Exactitude", pct(dep["accuracy"]), f"intervalle à 95 % : {pct(json.loads(dep['accuracy_ci95'])[0])} – "
                                         f"{pct(json.loads(dep['accuracy_ci95'])[1])}", "hl"),
    ("ROC AUC", fr(dep["roc_auc"], 4), "1 = classement parfait des photos"),
    ("Erreurs", f"{int(dep['errors'])} / {fr(n, 0)}", f"{cm[0][1]} chats vus comme chiens, {cm[1][0]} chiens vus comme chats",
     "orange"),
    ("Réponses sûres", pct(dep["confident_share"]), f"exactitude sur ces réponses : {pct(dep['confident_accuracy'], 2)}",
     "blue"),
])

section("Six modèles comparés", "Même échantillon, même test. Les barres fines donnent l'intervalle de confiance à 95 %.")
l, r = st.columns([1.35, 1], gap="medium")
k = card(l, "Exactitude sur le test", "Orange : le modèle déployé. Du CNN entraîné de zéro au transfert learning.")
t = table.copy().sort_values("accuracy", ascending=False)
lo_ax = 0.70                                       # axe de 70 % à 100 % : les écarts entre modèles restent lisibles
rows = ""
for _, row in t.iterrows():
    ci = json.loads(row["accuracy_ci95"])
    w = (row["accuracy"] - lo_ax) / (1 - lo_ax) * 100
    l_ci, r_ci = (ci[0] - lo_ax) / (1 - lo_ax) * 100, (ci[1] - lo_ax) / (1 - lo_ax) * 100
    color = ORANGE if row["deployed"] else DATA_BLUE
    tag = " · <b style='color:#B2470A'>déployé</b>" if row["deployed"] else ""
    rows += (f"<div class='srow wide'><div class='slab'>{row['label']}<span>{row['approach']}{tag}</span></div>"
             f"<div class='strack'><div class='sbar' style='left:0;width:{w:.1f}%;background:{color}'></div>"
             f"<div style='position:absolute;top:6px;height:2px;left:{l_ci:.1f}%;width:{max(r_ci - l_ci, .5):.1f}%;"
             f"background:{INK}'></div></div><div class='sval num'>{pct(row['accuracy'])}</div></div>")
k.markdown(rows + "<div class='c-cap' style='margin-top:.4rem'>Barres de 70 % à 100 %. Trait noir : intervalle de "
           "confiance à 95 % (bootstrap).</div>", unsafe_allow_html=True)

k = card(r, "Matrice de confusion", f"Modèle déployé ({dep['label']}), {fr(n, 0)} photos de test.")
cmd = pd.DataFrame([{"Réel": a, "Prédit": b, "n": cm[i][j]} for i, a in enumerate(["Chat", "Chien"])
                    for j, b in enumerate(["Chat", "Chien"])])
cmd["good"] = cmd["Réel"] == cmd["Prédit"]
rect = alt.Chart(cmd).mark_rect(cornerRadius=10, stroke="#FFFFFF", strokeWidth=6).encode(
    x=alt.X("Prédit:N", title="Prédit", axis=alt.Axis(orient="top", labelAngle=0, labelFontSize=14, ticks=False, domain=False)),
    y=alt.Y("Réel:N", title="Réel", axis=alt.Axis(labelFontSize=14, ticks=False, domain=False)),
    color=alt.condition(alt.datum.good, alt.value(DATA_BLUE), alt.value(ORANGE_RAMP[2])))
lab = alt.Chart(cmd).mark_text(fontSize=22, fontWeight=700).encode(
    x="Prédit:N", y="Réel:N", text="n:Q", color=alt.condition(alt.datum.good, alt.value("#FFFFFF"), alt.value(INK)))
k.altair_chart(chart_config(rect + lab, 260).configure_axis(grid=False), use_container_width=True)

show = table.copy()
show["Approche"] = show["approach"]
show["Exactitude"] = show["accuracy"].map(lambda v: pct(v, 2))
show["ROC AUC"] = show["roc_auc"].map(lambda v: fr(v, 4))
show["Perte log"] = show["log_loss"].map(lambda v: fr(v, 3))
show["Entrée"] = show["input_size"].map(lambda v: f"{int(v)} px")
show = show.sort_values("accuracy", ascending=False).rename(columns={"label": "Modèle"})
st.dataframe(show[["Modèle", "Approche", "Entrée", "Exactitude", "ROC AUC", "Perte log"]], hide_index=True,
             use_container_width=True)

section("Apprentissage et réglages")
l, r = st.columns(2, gap="medium")
if "mobilenet_v2_ft" in hist:
    h = hist["mobilenet_v2_ft"]
    k = card(l, "Fine-tuning de MobileNetV2", f"Étape 1 : tête seule ({h['stage1_epochs']} époques). Étape 2 : dégel "
                                             f"des derniers blocs ({h['stage2_epochs']} époques), taux d'apprentissage ÷ 50.")
    hh = h["history"]
    d = pd.DataFrame({"Époque": range(1, len(hh["accuracy"]) + 1), "Entraînement": hh["accuracy"],
                      "Validation": hh["val_accuracy"]}).melt("Époque", var_name="Jeu", value_name="Exactitude")
    line = alt.Chart(d).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("Époque:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Exactitude:Q", scale=alt.Scale(zero=False), axis=alt.Axis(labelExpr="format(datum.value * 100, '.0f') + ' %'")),
        color=alt.Color("Jeu:N", scale=alt.Scale(domain=["Entraînement", "Validation"], range=[DATA_BLUE, ORANGE]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Époque", "Jeu", alt.Tooltip("Exactitude:Q", format=".2%")])
    k.altair_chart(chart_config(line, 260), use_container_width=True)

trials = pd.DataFrame([{**tr, "Réseau": {"vgg16": "VGG16", "resnet50": "ResNet50", "inception_v3": "InceptionV3",
                                         "mobilenet_v2": "MobileNetV2"}[name]}
                       for name, s in search.items() for tr in s["random_search_trials"]])
k = card(r, "Recherche aléatoire d'hyperparamètres", f"{len(trials)} essais de tête de classification (16 par réseau) : "
                                                     "couche cachée, dropout, L2, taux d'apprentissage. Chaque point = un essai ; orange = tête retenue.")
trials["acc_fr"] = trials["val_accuracy"].map(lambda v: pct(v, 2))
trials["best"] = trials.groupby("Réseau")["val_loss"].transform("min") == trials["val_loss"]
pts = alt.Chart(trials).mark_circle(size=110, opacity=0.85).encode(
    x=alt.X("Réseau:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=14)),
    y=alt.Y("val_accuracy:Q", title="Exactitude en validation", scale=alt.Scale(zero=False),
            axis=alt.Axis(labelExpr="replace(format(datum.value * 100, '.1f'), '.', ',') + ' %'")),
    xOffset=alt.XOffset("jitter:Q"),
    color=alt.condition(alt.datum.best, alt.value(ORANGE), alt.value(DATA_BLUE)),
    tooltip=["Réseau", alt.Tooltip("units:Q", title="Neurones cachés"), alt.Tooltip("dropout:Q", title="Dropout"),
             alt.Tooltip("lr:Q", title="Taux d'apprentissage", format=".1e"), alt.Tooltip("acc_fr:N", title="Exactitude")]
).transform_calculate(jitter="random()")
k.altair_chart(chart_config(pts, 260), use_container_width=True)
footer()
