# 🛣️ Road Extraction - Segmentation de Routes par Images Satellites

Ce projet implémente une solution de Deep Learning pour l'extraction automatique de réseaux routiers à partir d'images satellites et aériennes. Il utilise une architecture de segmentation sémantique avancée pour générer des masques de routes précis, avec un cas d'usage spécifique sur la ville de Strasbourg.

## 📋 Aperçu du Projet

L'objectif est de traiter des images géoréférencées (TIFF) ou classiques (JPG) pour en extraire la structure routière. Le projet inclut l'exploration des données, l'entraînement (ou le chargement) d'un modèle, et la visualisation des résultats sous forme de cartes interactives.

### Fonctionnalités principales
* **Segmentation Sémantique** : Utilisation de réseaux de neurones convolutifs (CNN) pour la classification pixel par pixel (Route vs Non-Route).
* **Traitement Géospatial** : Gestion d'images géoréférencées (TIFF) pour projeter les prédictions sur des cartes réelles.
* **Visualisation Interactive** : Génération de cartes HTML pour superposer les routes détectées sur des fonds de carte réels.

## 🧠 Architecture Technique

Le cœur du système repose sur une architecture **U-Net**, standard de l'industrie pour la segmentation d'images.

* **Modèle** : U-Net
* **Encodeur (Backbone)** : ResNet-34 pré-entraîné sur ImageNet pour une extraction efficace des caractéristiques.
* **Librairie** : `segmentation-models-pytorch` et `torch`.
* **Entrée** : Images RGB (3 canaux).
* **Sortie** : Masque binaire (1 canal) représentant la probabilité de présence d'une route.

## 📂 Structure du Répertoire

```bash
road-extraction/
├── 📂 images/              # Images sources (ex: Strasbourg.jpg, Test_image.jpg)
├── 📂 maps/                # Cartes interactives générées (fichiers .html)
├── 📂 src/                 # Code source Python
│   ├── dataset.py          # Gestion du chargement des données et PyTorch Datasets
│   └── model.py            # Définitions relatives au modèle
├── Exploration.ipynb       # Notebook principal : démo, inférence et visualisation
├── courbe_loss.png         # Graphique de suivi de l'entraînement
├── UNet_20.pth             # (Fichier attendu) Poids du modèle entraîné
├── pyproject.toml          # Configuration du projet et dépendances
└── .python-version         # Version Python cible
```

## 🛠️ Installation et Pré-requis

Ce projet utilise Python. Il est recommandé d'utiliser un environnement virtuel (via `uv`, `conda` ou `venv`).

### Dépendances principales
Les bibliothèques suivantes sont nécessaires pour exécuter le notebook et les scripts :

* `torch` (PyTorch)
* `segmentation-models-pytorch`
* `opencv-python` (cv2)
* `matplotlib`
* `folium` (pour la génération de cartes)
* `rasterio` (pour les fichiers .tif géoréférencés)
* `numpy`

### Installation via pip

```bash
pip install torch segmentation-models-pytorch opencv-python matplotlib folium rasterio numpy
```

(Si vous utilisez `uv`, le fichier `uv.lock` présent dans le dépôt assure la reproductibilité de l'environnement).

## 🚀 Utilisation

Le point d'entrée principal pour tester le projet est le notebook Jupyter.

1. **Lancer le Notebook :**
   Ouvrez `Exploration.ipynb` dans Jupyter Lab ou VS Code.

2. **Workflow du Notebook :**
   * **Chargement des données :** Le script charge les images depuis le dossier `data/` ou `images/`.
   * **Initialisation du Modèle :** Chargement de l'architecture U-Net (ResNet34).
   * **Inférence :** Le modèle charge les poids pré-entraînés (ex: `UNet_20.pth`) et effectue une prédiction sur les images de test.
   * **Visualisation :** Les masques prédits sont affichés avec Matplotlib et exportés en cartes interactives dans le dossier `maps/`.

## 📊 Résultats

Le projet génère des visualisations permettant de comparer l'image satellite originale et le masque de route prédit.

Les fichiers de sortie dans le dossier `maps/` (ex: `Strasbourg.html`) peuvent être ouverts dans n'importe quel navigateur web pour explorer le réseau routier extrait superposé à une carte du monde.

## 👤 Auteur

Projet réalisé par Nathan Houel.