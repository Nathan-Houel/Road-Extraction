# 🛣️ Road Extraction - Road Segmentation from Satellite Images

This project implements a Deep Learning solution for the automatic extraction of road networks from satellite and aerial images. It uses an advanced semantic segmentation architecture to generate accurate road masks, with a specific use case on the city of Strasbourg.

## 📋 Project Overview

The goal is to process georeferenced images (TIFF) or standard images (JPG) to extract the road structure. The project includes data exploration, model training (or loading), and results visualization in the form of interactive maps.

### Main Features
* **Semantic Segmentation** : Uses Convolutional Neural Networks (CNN) for pixel-by-pixel classification (Road vs. Non-Road).
* **Geospatial Processing** : Handles georeferenced images (TIFF) to project predictions onto real-world maps.
* **Interactive Visualization** : Generates HTML maps to overlay detected roads on real map backgrounds.

## 🧠 Technical Architecture

The core of the system relies on a **U-Net** architecture, an industry standard for image segmentation.

* **Model** : U-Net
* **Encoder (Backbone)** : ResNet-34 pre-trained on ImageNet for efficient feature extraction.
* **Library** : `segmentation-models-pytorch` and `torch`.
* **Input** : RGB Images (3 channels).
* **Output** : Binary mask (1 channel) representing the probability of a road being present.

## 📂 Directory Structure

```bash
road-extraction/
├── 📂 images/              # Source images (e.g., Strasbourg.jpg, Test_image.jpg)
├── 📂 maps/                # Generated interactive maps (.html files)
├── 📂 src/                 # Python source code
│   ├── dataset.py          # Data loading management and PyTorch Datasets
│   └── model.py            # Model-related definitions
├── Exploration.ipynb       # Main notebook: demo, inference, and visualization
├── courbe_loss.png         # Training loss plot
├── UNet_20.pth             # (Expected file) Trained model weights
├── pyproject.toml          # Project configuration and dependencies
└── .python-version         # Target Python version
```

## 🛠️ Installation and Prerequisites

This project uses Python. It is recommended to use a virtual environment (via uv, conda, or venv).

### Main Dependencies
The following libraries are required to run the notebook and scripts :

* `torch` (PyTorch)
* `segmentation-models-pytorch`
* `opencv-python` (cv2)
* `matplotlib`
* `folium` (for map generation)
* `rasterio` (for georeferenced .tif files)
* `numpy`

### Installation via pip

```bash
pip install torch segmentation-models-pytorch opencv-python matplotlib folium rasterio numpy
```

(If you are using uv, the uv.lock file present in the repository ensures environment reproducibility).

## 🚀 Usage

The main entry point to test the project is the Jupyter notebook.

1. **Launch the Notebook :**
   Open `Exploration.ipynb` in Jupyter Lab or VS Code.

2. **Notebook Workflow :**
   * **Data Loading :** The script loads images from the `data/` or `images/` folder.
   * **Model Initialization :** Loads the U-Net architecture (ResNet34).
   * **Inference :** The model loads pre-trained weights (e.g., `UNet_20.pth`) and performs prediction on test images.
   * **Visualization :** Predicted masks are displayed using Matplotlib and exported as interactive maps in the `maps/` folder.

## 📊 Results

The project generates visualizations allowing for a comparison between the original satellite image and the predicted road mask.

The output files in the `maps/` folder (e.g., `Strasbourg.html`) can be opened in any web browser to explore the extracted road network overlaid on a world map.

## 👤 Author

Project created by Nathan Houel.