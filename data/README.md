# Données

**Source** : jeu Kaggle [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) (25 000 photos étiquetées, issues
du projet Asirra de Microsoft Research et de Petfinder.com).

Kaggle n'étant pas joignable depuis l'environnement d'entraînement, les images viennent du miroir GitHub
[laxmimerit/dog-cat-full-dataset](https://github.com/laxmimerit/dog-cat-full-dataset), épinglé au commit
`b2d738611a3daada177ea755c11b4be14726d38c` (24 989 photos : les 25 000 du jeu Kaggle moins 11 fichiers).

```bash
make data   # clone le miroir, se place sur le commit épinglé, contrôle et découpe les images
```

**Les images ne sont pas republiées** dans ce dépôt (licence du jeu Kaggle). Le dépôt contient seulement :

| Fichier | Contenu |
|---|---|
| `processed/manifest.csv` | les 12 000 photos de l'échantillon : nom du fichier, classe, échantillon (train / val / test), taille, empreintes SHA-256 et pHash |
| `../reports/data_quality.json` | contrôle qualité : fichiers illisibles, doublons exacts et quasi identiques, effectifs |

Les dossiers `raw/`, `cache/` et `features/` sont exclus de Git (`.gitignore`).
