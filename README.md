# Plant 🌱 Disease 🐛 Detection 🔎

Plant Disease Detection is an innovative machine learning project that harnesses the power of Convolutional Neural Networks (CNN) and deep learning techniques to identify and classify diseases in plants. The primary objective is to offer farmers and agricultural experts a valuable tool for swift plant health diagnosis, facilitating timely intervention and minimizing the risk of crop loss.

## Project Structure 📂

The project comprises essential components:

- `Plant_Disease_Detection.ipynb`: Jupyter Notebook with the code for model training.
- `main_app.py`: Streamlit web application for plant disease prediction.
- `train_model.py` / `dataset.py`: Training and dataset class discovery.
- `plant_disease_model.keras` / `class_names.json`: Trained model and label mapping.
- `plant_disease_model.h5`: Original three-class model (legacy).
- `requirements.txt`: List of necessary Python packages.

## Installation 🚀

To run the project locally, follow these steps:

1. **Clone the repository:**

```bash
git clone https://github.com/anivish2004/Plant-Disease-Detection.git
```

2. Navigate to the project directory:

```bash
cd Plant-Disease-Detection
```

3. **Install the required packages:**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit web application:**

```bash
streamlit run main_app.py
```

## Usage 🌿

Once the application is running, open your web browser and navigate to [http://localhost:8501](http://localhost:8501). Upload an image of a plant leaf, and the system will predict if it is affected by any disease.

## Model Training 🧠

The training pipeline discovers **all 38 classes** (54,305 images) in PlantVillage's
`raw/color` directory, including healthy leaves. Class names are sorted and saved
with the trained model so inference uses the same label order as training.

The expanded model was trained on this dataset for 30 epochs and achieved
**96.86% held-out test accuracy** on 8,129 images (seed 42). Mean per-class test
accuracy was **96.24%**. See `class_names.json` and `test_class_metrics.json`
for the saved results; accuracy varies by class.

Use Python 3.11 or 3.12 and install `requirements.txt`. On this machine the default
dataset is `~/Desktop/PlantVillage-Dataset/raw/color`; override it as needed:

```bash
python3 train_model.py --inspect
python3 train_model.py --dataset /path/to/PlantVillage-Dataset/raw/color --epochs 30
```

Alternatively, run `Plant_Disease_Detection.ipynb` from the project directory.
Both entry points use the same trainer. Images stream in batches, with per-class
70% training / 15% validation / 15% test splits. A frozen ImageNet-pretrained MobileNetV2 encoder extracts image features once,
then a new classification head learns all classes. RGB inputs are resized to
128 × 128 and normalized inside the model to [-1, 1], matching the app.
Inverse-frequency class weights give the smaller classes weight during training.
The initial run downloads the encoder weights; they are included in the exported model. Early stopping restores the best validation weights.

Training exports:

- `plant_disease_model.keras`: expanded classifier used by the app.
- `class_names.json`: ordered labels, preprocessing, class counts, and test metrics.
- `training_history.json`: training and validation curves.
- `test_class_metrics.json`: held-out accuracy and image count for each class.
- `training_head_checkpoint.keras` and `training_log.csv`: best classification-head checkpoint and epoch log.
  The head checkpoint alone is not the full image classifier.

The original `plant_disease_model.h5` contains only three outputs. The expanded `.keras` classifier was trained separately; changing label names
cannot expand the original weights. The app requires the new model and its matching metadata. Test accuracy
is measured on a held-out PlantVillage split and does not establish field accuracy.

## Web Application 🌐

The web application (`main_app.py`) empowers users to interact with the trained model. Upload plant images, and the application provides real-time predictions regarding the health of the plant.

## Requirements 🛠️

Dependencies are listed in `requirements.txt`: TensorFlow 2.16 or newer, NumPy,
Streamlit, Pillow, and Matplotlib. Training on CPU can take several hours.

## Compare four models

Run the original repository CNN (retrained for 38 classes), MobileNetV2,
EfficientNetB0, and DenseNet121 on the same dataset splits:

```bash
.venv/bin/python compare_models.py
```

All models use 128 × 128 RGB images, the same stratified splits, class weights,
and up to 30 epochs. The original CNN keeps its Conv32/Pool3/Conv16/Pool2/Flatten/
Dense8 structure and trains from scratch; the other three use frozen pretrained
encoders with a Dense256/Dropout0.3 head. This is a comparison of these training
approaches at a shared resolution, not a full fine-tuning benchmark.

`Model_Comparison.ipynb` shows overall and per-class results, confusion matrices,
and training curves. The `comparison/` folder saves the split manifest with
image hashes, models, mappings, predictions, a CSV/JSON summary, a chart, and
`comparison_report.md`. Cached compatible runs can be resumed without retraining
completed models. The preferred model is selected by **validation macro F1**;
test scores are reported separately.

In Streamlit, choose **Model comparison** in the sidebar to view results, or
choose a trained model under **Predict** to try it on a leaf image.

Measured comparison (same 8,129 test images):

| Model | Test accuracy | Test macro F1 |
|---|---:|---:|
| Original repository CNN | 51.27% | 44.21% |
| MobileNetV2 | 96.86% | 96.06% |
| EfficientNetB0 | 97.44% | 96.72% |
| DenseNet121 | 97.59% | 96.88% |

DenseNet121 also had the highest validation macro F1 and is the preferred model
for this experiment. The original CNN stopped at 22 epochs; MobileNetV2 ran 30,
EfficientNetB0 20, and DenseNet121 27, using the shared early-stopping rule.
See [the full comparison report](comparison/comparison_report.md) for measured
runtime, inference speed, artifact sizes, and the experiment's scope.
