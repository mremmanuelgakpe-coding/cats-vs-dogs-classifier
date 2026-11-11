"""Figures du README (palette du portfolio : bleu de données, orange pour le modèle déployé).

Usage : PYTHONPATH=src python -m catdog.plots
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .data import load_manifest  # noqa: E402

BLUE, ORANGE, INK, MUTED, LIGHT = "#3B5C99", "#F4721E", "#1E2A3B", "#5E6878", "#FBC9A6"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#D9CFC4",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlecolor": INK})


def pct(x, nd=1):
    return f"{100 * x:.{nd}f} %".replace(".", ",")


def comparison(t):
    t = t.sort_values("accuracy")
    ci = np.array([json.loads(v) for v in t["accuracy_ci95"]])
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [ORANGE if d else BLUE for d in t["deployed"]]
    ax.barh(t["label"], t["accuracy"], color=colors, height=0.6)
    ax.errorbar(t["accuracy"], t["label"], xerr=[t["accuracy"] - ci[:, 0], ci[:, 1] - t["accuracy"]], fmt="none",
                ecolor=INK, capsize=3, lw=1)
    for y, (acc, hi) in enumerate(zip(t["accuracy"], ci[:, 1])):
        ax.text(hi + 0.002, y, pct(acc), va="center", color=INK, fontweight="bold")
    ax.set_xlim(max(0, ci[:, 0].min() - 0.03), 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
    ax.set_title("Exactitude sur 2 000 photos de test (IC 95 %)", loc="left")
    fig.savefig(config.FIGURES_DIR / "model_comparison.png")
    plt.close(fig)


def confusion(cm, label):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    colors = np.array([[BLUE, LIGHT], [LIGHT, BLUE]])
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 0.96, 0.96, color=colors[i, j]))
            ax.text(j + 0.48, 1 - i + 0.48, f"{cm[i, j]:,}".replace(",", " "), ha="center", va="center", fontsize=20,
                    fontweight="bold", color="white" if i == j else INK)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.48, 1.48], ["Chat", "Chien"])
    ax.set_yticks([1.48, 0.48], ["Chat", "Chien"])
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Matrice de confusion ({label})", loc="left", fontsize=12)
    fig.savefig(config.FIGURES_DIR / "confusion_matrix.png")
    plt.close(fig)


def finetune_curves():
    h = json.loads((config.REPORTS_DIR / "history_mobilenet_v2_ft.json").read_text())
    hh, s1 = h["history"], h["stage1_epochs"]
    ep = np.arange(1, len(hh["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, title in [(axes[0], "accuracy", "Exactitude"), (axes[1], "loss", "Perte (entropie croisée)")]:
        ax.plot(ep, hh[key], "-o", color=BLUE, label="Entraînement", ms=4)
        ax.plot(ep, hh[f"val_{key}"], "-o", color=ORANGE, label="Validation", ms=4)
        ax.axvline(s1 + 0.5, color=MUTED, ls="--", lw=1)
        ax.text(s1 + 0.6, ax.get_ylim()[1], " dégel des derniers blocs", va="top", color=MUTED, fontsize=10)
        ax.set_xlabel("Époque")
        ax.set_title(title, loc="left", fontsize=12)
        if key == "accuracy":
            ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
    axes[0].legend(frameon=False)
    fig.suptitle("Fine-tuning de MobileNetV2", x=0.02, y=1.05, ha="left", fontweight="bold", color=INK)
    fig.savefig(config.FIGURES_DIR / "finetune_curves.png")
    plt.close(fig)


def random_search(search):
    names = {"vgg16": "VGG16", "resnet50": "ResNet50", "inception_v3": "InceptionV3", "mobilenet_v2": "MobileNetV2"}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    rng = np.random.default_rng(0)
    for i, (k, s) in enumerate(search.items()):
        acc = [t["val_accuracy"] for t in s["random_search_trials"]]
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(acc)), acc, color=BLUE, alpha=0.75, s=36)
        ax.scatter([i], [s["random_search_best"]["val_accuracy"]], color=ORANGE, s=90, zorder=3, edgecolor="white")
    ax.set_xticks(range(len(search)), [names[k] for k in search])
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.1f} %")
    ax.set_ylabel("Exactitude en validation")
    ax.set_title("Random search : 16 têtes testées par réseau (orange : retenue)", loc="left", fontsize=12)
    fig.savefig(config.FIGURES_DIR / "random_search.png")
    plt.close(fig)


def reliability(name):
    y = load_manifest("test")["y"].to_numpy()
    p = np.load(config.REPORTS_DIR / "predictions" / f"{name}_test.npy")
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    d = pd.DataFrame({"b": idx, "p": p, "y": y}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    ax.plot(d["p"], d["y"], "-", color=ORANGE, lw=1.2, alpha=0.6)
    ax.scatter(d["p"], d["y"], s=20 + 300 * np.sqrt(d["n"] / d["n"].max()), color=ORANGE, zorder=3)
    for _, r in d.iterrows():
        ax.annotate(f"{int(r['n'])}", (r["p"], r["y"]), textcoords="offset points", xytext=(6, -12), fontsize=9,
                    color=MUTED)
    ax.set_xlabel("Probabilité « chien » prédite")
    ax.set_ylabel("Part réelle de chiens")
    ax.set_title("Calibration (test) : nombre de photos par tranche", loc="left", fontsize=11)
    fig.savefig(config.FIGURES_DIR / "calibration.png")
    plt.close(fig)


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    m = json.loads((config.REPORTS_DIR / "metrics.json").read_text())
    t = pd.read_csv(config.REPORTS_DIR / "model_comparison.csv")
    comparison(t)
    confusion(m["deployed_confusion"], config.BACKBONE_LABELS[m["deployed"]])
    finetune_curves()
    random_search(json.loads((config.REPORTS_DIR / "transfer_search.json").read_text()))
    reliability(m["deployed"])


if __name__ == "__main__":
    main()
