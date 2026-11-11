"""Chemins, graine, sources des données et des poids pré-entraînés (avec empreintes SHA-256)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"
WEIGHTS_DIR = ROOT / "weights"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = 42
CLASSES = ["cat", "dog"]          # 0 = chat, 1 = chien
CLASSES_FR = {"cat": "Chat", "dog": "Chien"}

# Copie GitHub du jeu Kaggle « Dogs vs. Cats » (25 000 photos étiquetées, Asirra / Microsoft Research).
# Kaggle n'étant pas joignable depuis l'environnement d'entraînement, on part de ce miroir, épinglé à un commit.
DATA_REPO = "https://github.com/laxmimerit/dog-cat-full-dataset.git"
DATA_COMMIT = "b2d738611a3daada177ea755c11b4be14726d38c"
DATA_SRC = RAW_DIR / "src_repo" / "data"

# Échantillon stratifié (temps de calcul sur CPU) : 12 000 photos uniques, 6 000 par classe.
SAMPLE_PER_CLASS = 6000
SPLIT = {"train": 0.6667, "val": 0.1667, "test": 0.1666}   # ≈ 8 000 / 2 000 / 2 000

# Poids ImageNet officiels Keras, téléchargés depuis leurs publications GitHub (sans tête de classification).
WEIGHTS = {
    "vgg16": ("https://github.com/fchollet/deep-learning-models/releases/download/v0.1/"
              "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
              "bfe5187d0a272bed55ba430631598124cff8e880b98d38c9e56c8d66032abdc1"),
    "resnet50": ("https://github.com/keras-team/keras-applications/releases/download/resnet/"
                 "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                 "66c8b43daff3fcc15bc4f30e3d2a167e21a14d9c9598a5394e5516471f4af504"),
    "inception_v3": ("https://github.com/fchollet/deep-learning-models/releases/download/v0.5/"
                     "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
                     "9617109a16463f250180008f9818336b767bdf5164315e8cd5761a8c34caa62a"),
    "mobilenet_v2": ("https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/"
                     "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
                     "f8aff69536bd77a692c594f559c798c19bf7f3f36668fc9fa00b21c6aab4797c"),
    # MobileNetV2 complet (1 000 classes ImageNet) : garde-fou « est-ce bien un chat ou un chien ? »
    "mobilenet_v2_top": ("https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/"
                         "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5",
                         "3e195a2857356cfc092cbbb460beb2a5bce279015d7792598b8d3d9e451902e3"),
}

# Taille d'entrée de chaque réseau pour l'extraction de caractéristiques (compromis précision / temps CPU)
BACKBONES = {"vgg16": 160, "resnet50": 224, "inception_v3": 224, "mobilenet_v2": 224}
BACKBONE_LABELS = {"vgg16": "VGG16", "resnet50": "ResNet50", "inception_v3": "InceptionV3",
                   "mobilenet_v2": "MobileNetV2", "cnn_scratch": "CNN maison", "mobilenet_v2_ft": "MobileNetV2 affiné"}

CNN_SIZE = 128       # CNN entraîné de zéro
FT_SIZE = 160        # fine-tuning de MobileNetV2 (modèle déployé)
UNSURE_BAND = (0.2, 0.8)  # probabilité « chien » entre ces bornes : réponse « incertain »
