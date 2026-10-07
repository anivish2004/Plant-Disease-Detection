"""Streamlit inference using the class mapping exported during training."""
import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from model_architectures import MODEL_NAMES

PROJECT_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_classifier(directory, model_modified, metadata_modified):
    model_path = Path(directory) / "plant_disease_model.keras"
    metadata_path = Path(directory) / "class_names.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            "Train the expanded classifier first: python3 train_model.py. "
            "It saves plant_disease_model.keras and class_names.json. "
            "The bundled .h5 model supports only the original three classes."
        )
    metadata = json.loads(metadata_path.read_text())
    model = tf.keras.models.load_model(model_path, compile=False)
    classes = metadata["class_names"]
    if model.output_shape[-1] != len(classes):
        raise ValueError("Model and class mapping differ. Use files from the same training run.")
    size = metadata["image_size"]
    if model.input_shape[1:] != (size, size, 3):
        raise ValueError("Model input dimensions do not match the saved metadata.")
    if metadata.get("color_mode") != "RGB" or metadata.get("normalization") not in {
        "model_rescaling_1_over_127_5_minus_1", "model_mobilenetv3_preprocessing",
        "model_efficientnet_preprocessing", "model_imagenet_rgb_normalization",
        "model_rescaling_1_over_255",
    }:
        raise ValueError("Unsupported preprocessing metadata. Retrain with train_model.py.")
    return model, classes, size


st.title("Plant Disease Detection")
view = st.sidebar.radio("View", ["Predict", "Model comparison"])
comparison_dir = PROJECT_DIR / "comparison"
if view == "Model comparison":
    results_path = comparison_dir / "comparison_results.json"
    if not results_path.is_file():
        st.info("The comparison results will appear when model training finishes.")
    else:
        report = json.loads(results_path.read_text())
        st.subheader("Models on the same PlantVillage dataset")
        st.caption(f"{len(report['results'])} models completed; identical training, validation, and test images.")
        rows = [{"Model": result["model"],
                 "Test accuracy": f"{result['test']['accuracy']:.2%}",
                 "Test macro F1": f"{result['test']['macro_f1']:.2%}",
                 "Validation macro F1": f"{result['validation']['macro_f1']:.2%}",
                 "Training minutes": round(result["training_seconds"] / 60, 2),
                 "Model size (MB)": round(result["model_size_mb"], 2)}
                for result in report["results"]]
        st.dataframe(rows, hide_index=True)
        st.write("Preferred model by validation macro F1: " + report["preferred_model_by_validation_macro_f1"])
        chart = comparison_dir / "comparison_chart.png"
        if chart.is_file():
            st.image(str(chart))
        st.caption("Results are from one seeded experiment on held-out PlantVillage images. Training time excludes weight downloads; inference is measured in batches of 32.")
    st.stop()

st.caption("Upload one leaf image to see predictions from all four models across 38 plant disease and healthy-leaf classes.")
plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if plant_image is not None:
    try:
        image = Image.open(plant_image).convert("RGB")
        # Use the training decoder for inference; JPEG decoders can differ.
        decoded_image = tf.io.decode_image(
            plant_image.getvalue(), channels=3, expand_animations=False)
    except (UnidentifiedImageError, OSError, tf.errors.OpError):
        st.error("The uploaded file could not be read as an image.")
        st.stop()
    st.image(image)
    rows = []
    resized_images = {}
    with st.spinner("Running all four models..."):
        for name in MODEL_NAMES:
            directory = comparison_dir / name
            try:
                model_path = directory / "plant_disease_model.keras"
                metadata_path = directory / "class_names.json"
                model, class_names, image_size = load_classifier(
                    str(directory), model_path.stat().st_mtime_ns,
                    metadata_path.stat().st_mtime_ns)
                # Match training resize; each model contains its own normalization.
                if image_size not in resized_images:
                    pixels = tf.image.resize(decoded_image, [image_size, image_size])
                    resized_images[image_size] = tf.expand_dims(pixels, 0)
                prediction = model(resized_images[image_size], training=False).numpy()[0]
                index = int(np.argmax(prediction))
                plant, condition = class_names[index].split("___", 1)
                rows.append({
                    "Model": name,
                    "Plant": plant.replace("_", " "),
                    "Condition": condition.replace("_", " ").strip(),
                    "Confidence": f"{float(prediction[index]) * 100}%",
                })
            except (ValueError, OSError, KeyError, tf.errors.OpError) as error:
                st.error(f"{name} could not produce a prediction: {error}")
    if rows:
        st.subheader("Predictions for your image")
        st.dataframe(rows, hide_index=True)
