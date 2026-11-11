# Fiche modèle : Cats vs Dogs v1.0.0

| Élément | Détail |
|---|---|
| Tâche | Classification binaire d'images : probabilité que la photo montre un chien (sinon un chat) |
| Modèle | MobileNetV2 pré-entraîné sur ImageNet (gelé) + tête Keras (Dropout 0,4, Dense 512 ReLU, Dense 1 sigmoïde) |
| Entrée | Image RVB 224 × 224, pixels 0-255 ; prétraitement inclus dans le graphe ONNX |
| Données | Kaggle Dogs vs. Cats (Asirra), 12 000 photos uniques : 8 000 entraînement, 2 000 validation, 2 000 test |
| Réglages | Random search (16 essais) sur la validation ; règle de choix fixée avant le test |
| Format | ONNX (11,5 Mo), empreinte SHA-256 dans `models/model.meta.json` |
| Garde-fou | MobileNetV2 complet (1 000 classes ImageNet) : signale les images sans chat ni chien |

## Performances (test, 2 000 photos)

| Exactitude | IC 95 % | ROC AUC | Perte log | Erreurs | Réponses sûres | Exactitude des réponses sûres |
|---|---|---|---|---|---|---|
| 98,9 % | 98,4 – 99,4 % | 0,9991 | 0,032 | 22 (11 + 11) | 99,0 % | 99,5 % |

Temps de réponse : ≈ 6 ms par photo sur processeur (ONNX Runtime, 2 threads).

## Usage prévu

Démonstration de portfolio : classer des photos de chats et de chiens et expliquer la décision. Ce n'est pas un outil
d'identification d'animaux (race, individu) ni un système de modération.

## Limites et risques

- Deux classes : une image hors sujet reçoit quand même une réponse ; le garde-fou la signale (seuil fixé pour laisser
  passer 99 % des vraies photos de chats et chiens, son taux de détection des images hors sujet n'est pas mesuré sur un
  jeu dédié).
- Photos d'amateurs issues de Petfinder (fin des années 2000) : la représentativité pour d'autres sources (téléphones
  récents, dessins, caméras de surveillance) n'est pas garantie.
- Aucune donnée personnelle n'est traitée ; les photos envoyées à l'API ou à la démo ne sont pas conservées.

## Réentraînement

`make data weights train evaluate` régénère l'échantillon, les caractéristiques, les modèles, l'export ONNX et les
figures (graine 42).
