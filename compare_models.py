"""Compare four ImageNet encoders on identical PlantVillage splits."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from dataset import DEFAULT_DATASET, discover_classes, split_dataset
from model_architectures import MODEL_NAMES, build_classifier
from train_model import PROJECT_DIR, make_dataset


def classification_metrics(actual, predicted, names):
    import numpy as np
    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(matrix, (actual, predicted), 1)
    true_positive = np.diag(matrix)
    support = matrix.sum(axis=1)
    precision = np.divide(true_positive, matrix.sum(axis=0),
                          out=np.zeros(len(names)), where=matrix.sum(axis=0) != 0)
    recall = np.divide(true_positive, support, out=np.zeros(len(names)), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(names)), where=(precision + recall) != 0)
    summary = {"accuracy": float(true_positive.sum() / matrix.sum()),
               "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
               "macro_f1": float(f1.mean()), "weighted_f1": float(np.average(f1, weights=support))}
    detail = {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                     "f1": float(f1[i]), "support": int(support[i])}
              for i, name in enumerate(names)}
    return summary, detail, matrix


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def run_model(name, classes, splits, config, output):
    import numpy as np
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["seed"])
    destination = output / name
    destination.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    cached_result = destination / "results.json"
    if cached_result.is_file():
        previous = json.loads(cached_result.read_text())
        if previous.get("protocol_sha256") == signature and (destination / "plant_disease_model.keras").is_file():
            feature_meta = destination / "features/metadata.json"
            if feature_meta.is_file():
                timings = json.loads(feature_meta.read_text())["extraction_seconds"]
                previous["training_seconds"] = timings.get("train", 0) + timings.get("validation", 0) + previous["head_training_seconds"]
                write_json(cached_result, previous)
            print(f"Reusing completed {name} run", flush=True)
            return previous
    print(f"\n=== {name} ===", flush=True)
    model, normalization = build_classifier(name, len(classes), config["image_size"])
    feature_seconds = {}
    if name == "OriginalCNN":
        learner = model
        learner.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        learning_data = {split: make_dataset(paths, labels, config["image_size"],
                                            config["batch_size"], training=split == "train", seed=config["seed"])
                         for split, (paths, labels) in splits.items()}
        def evaluation_data(split):
            return make_dataset(*splits[split], config["image_size"], config["batch_size"])
    else:
        encoder = model.layers[0]
        feature_dir = destination / "features"
        feature_dir.mkdir(exist_ok=True)
        feature_meta = feature_dir / "metadata.json"
        feature_record = json.loads(feature_meta.read_text()) if feature_meta.is_file() else {}
        features = {}
        feature_seconds = feature_record.get("extraction_seconds", {}) if feature_record.get("protocol_sha256") == signature else {}
        class Progress(tf.keras.callbacks.Callback):
            def on_predict_batch_end(self, batch, logs=None):
                if (batch + 1) % 100 == 0:
                    print(f"{name}: {batch + 1} feature batches processed", flush=True)
        for split, (paths, labels) in splits.items():
            feature_path = feature_dir / f"{split}.npy"
            if feature_path.is_file() and feature_record.get("protocol_sha256") == signature:
                features[split] = np.load(feature_path)
            else:
                print(f"{name}: extracting {split} features ({len(paths)} images)", flush=True)
                start = time.perf_counter()
                data = make_dataset(paths, labels, config["image_size"], config["batch_size"])
                features[split] = encoder.predict(data, verbose=2, callbacks=[Progress()])
                feature_seconds[split] = time.perf_counter() - start
                np.save(feature_path, features[split])
                write_json(feature_meta, {"protocol_sha256": signature, "extraction_seconds": feature_seconds})
            if len(features[split]) != len(paths):
                raise ValueError(f"Feature count mismatch for {name}/{split}")
        learner = tf.keras.Sequential([tf.keras.layers.Input(shape=(features["train"].shape[1],)),
                                      *model.layers[1:]], name="classification_head")
        learner.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        def evaluation_data(split):
            data = tf.data.Dataset.from_tensor_slices(
                (features[split], np.asarray(splits[split][1], dtype="int32")))
            return data.batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)
        train_features = tf.data.Dataset.from_tensor_slices(
            (features["train"], np.asarray(splits["train"][1], dtype="int32")))
        train_features = train_features.shuffle(len(splits["train"][0]), seed=config["seed"])
        learning_data = {"train": train_features.batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE),
                         "validation": evaluation_data("validation")}
    counts = np.bincount(splits["train"][1], minlength=len(classes))
    weights = {i: float(sum(counts) / (len(classes) * count)) for i, count in enumerate(counts)}
    start = time.perf_counter()
    class TrainingProgress(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if name == "OriginalCNN" and (batch + 1) % 200 == 0:
                print(f"OriginalCNN: batch {batch + 1}, accuracy={logs['accuracy']:.4f}, loss={logs['loss']:.4f}", flush=True)

    history = learner.fit(learning_data["train"], validation_data=learning_data["validation"],
                          epochs=config["epochs"], class_weight=weights, verbose=2,
                          callbacks=[TrainingProgress(), tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                                      restore_best_weights=True),
                                     tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2),
                                     tf.keras.callbacks.ModelCheckpoint(str(destination / "training_checkpoint.keras"), save_best_only=True),
                                     tf.keras.callbacks.CSVLogger(str(destination / "training_log.csv")),
                                     tf.keras.callbacks.TerminateOnNaN()])
    head_seconds = time.perf_counter() - start
    # Choose by validation macro F1. Test scores never select the preferred model.
    validation_predictions = np.argmax(learner.predict(evaluation_data("validation"), verbose=0), axis=1)
    validation, _, _ = classification_metrics(np.asarray(splits["validation"][1]), validation_predictions, list(classes))
    test_probabilities = learner.predict(evaluation_data("test"), verbose=0)
    if not np.isfinite(test_probabilities).all():
        raise ValueError(f"Non-finite predictions: {name}")
    predicted = np.argmax(test_probabilities, axis=1)
    test, detail, matrix = classification_metrics(np.asarray(splits["test"][1]), predicted, list(classes))
    loss = learner.evaluate(evaluation_data("test"), verbose=0, return_dict=True)["loss"]
    test["loss"] = float(loss)
    inference_images, _ = next(iter(make_dataset(*splits["test"], config["image_size"], 32)))
    @tf.function
    def infer(images):
        return model(images, training=False)
    infer(inference_images).numpy()  # Warm up tracing and kernels before timing.
    durations = []
    for _ in range(5):
        start = time.perf_counter()
        infer(inference_images).numpy()
        durations.append(time.perf_counter() - start)
    model_path = destination / "plant_disease_model.keras"
    model.save(model_path)
    metadata = {"class_names": list(classes), "image_size": config["image_size"],
                "color_mode": "RGB", "normalization": normalization, "seed": config["seed"],
                "encoder": "original_CNN_from_scratch" if name == "OriginalCNN" else f"{name}_ImageNet_frozen", "test_metrics": test,
                "class_counts": {key: len(value) for key, value in classes.items()},
                "split_counts": {key: len(value[0]) for key, value in splits.items()},
                "balanced_class_weights": True}
    write_json(destination / "class_names.json", metadata)
    write_json(destination / "training_history.json", history.history)
    write_json(destination / "test_class_metrics.json", detail)
    np.save(destination / "test_predictions.npy", predicted)
    np.save(destination / "confusion_matrix.npy", matrix)
    results = {"model": name, "protocol_sha256": signature, "validation": validation, "test": test,
               "epochs_completed": len(history.history["loss"]),
               "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
               "parameters": int(model.count_params()),
               "trainable_parameters": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
               "feature_extraction_seconds": float(sum(feature_seconds.values())),
               "head_training_seconds": head_seconds,
               "training_seconds": float(feature_seconds.get("train", 0) + feature_seconds.get("validation", 0) + head_seconds),
               "batch32_inference_ms_per_image": float(np.median(durations) * 1000 / len(inference_images)),
               "model_size_mb": model_path.stat().st_size / 1e6}
    write_json(cached_result, results)
    print(f"{name} test accuracy={test['accuracy']:.4f}, macro F1={test['macro_f1']:.4f}", flush=True)
    del learning_data, learner, model
    tf.keras.backend.clear_session()
    gc.collect()
    return results


def generate_report(results, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = max(results, key=lambda result: result["validation"]["macro_f1"])["model"]
    write_json(output / "comparison_results.json", {"protocol": config,
                                                   "preferred_model_by_validation_macro_f1": best,
                                                   "results": results})
    columns = ["model", "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1",
               "validation_macro_f1", "training_seconds", "batch32_inference_ms_per_image",
               "model_size_mb", "parameters", "epochs_completed"]
    with (output / "comparison_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow({"model": result["model"],
                             **{f"test_{key}": result["test"][key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")},
                             "validation_macro_f1": result["validation"]["macro_f1"],
                             **{key: result[key] for key in columns[6:]}})
    names = [result["model"] for result in results]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for offset, metric in enumerate(("accuracy", "macro_f1")):
        axes[0].bar([i + offset * .4 for i in range(len(results))],
                    [100 * result["test"][metric] for result in results], width=.4, label=metric.replace('_', ' ').title())
    axes[0].set_xticks([i + .2 for i in range(len(results))], names, rotation=25, ha="right")
    axes[0].set_ylabel("Test score (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend()
    for axis, metric, label in [(axes[1], "training_seconds", "Feature extraction + training (minutes)"),
                                (axes[2], "batch32_inference_ms_per_image", "Batch-32 inference (ms/image)")]:
        values = [result[metric] / (60 if metric == "training_seconds" else 1) for result in results]
        axis.bar(names, values)
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output / "comparison_chart.png", dpi=160)
    plt.close(fig)
    report = ["# Four-model comparison", "", f"Preferred model by validation macro F1: **{best}**.", "",
              f"All models use the same {len(config['class_names'])}-class PlantVillage raw/color dataset and stratified image splits.",
              f"RGB images are resized to {config['image_size']} × {config['image_size']} for all models.",
              (f"OriginalCNN preserves the repository Conv32/Pool3/Conv16/Pool2/Flatten/Dense8 structure and trains from scratch, with {len(config['class_names'])} output classes. The prior input resolution was 256; this experiment uses the common resolution above."
               if "OriginalCNN" in names else
               "MobileNetV3Large replaces the original CNN baseline, which achieved 51.27% test accuracy. The historical baseline artifacts remain in OriginalCNN/."),
              ("The pretrained models use frozen ImageNet encoders and a Dense(256)/Dropout(0.3) classification head."),
              f"Maximum {config['epochs']} epochs, Adam(0.001), inverse-frequency class weights, and identical early-stopping policies.", "",
              "| Model | Test accuracy | Test macro F1 | Validation macro F1 | Training minutes | Batch-32 ms/image | Size MB |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        report.append(f"| {r['model']} | {r['test']['accuracy']:.2%} | {r['test']['macro_f1']:.2%} | {r['validation']['macro_f1']:.2%} | {r['training_seconds']/60:.2f} | {r['batch32_inference_ms_per_image']:.2f} | {r['model_size_mb']:.2f} |")
    report += ["", "![Comparison](comparison_chart.png)", "",
               "Training timings are from this CPU and include train/validation feature extraction plus fitting, excluding test extraction, weight downloads, and model construction. Inference timing excludes image decoding and resizing.",
               "This is one seeded comparison at a common resolution, not a full fine-tuning or native-resolution benchmark.",
               f"The test split contains {config['split_counts']['test']:,} images; scores describe held-out PlantVillage images, not field performance.",
               "Model selection uses validation macro F1. Upload one image in the app to see predictions from all four current models together.",
               "Saved split_manifest.csv fixes image membership and class order, and includes each image SHA256 for cache integrity."]
    (output / "comparison_report.md").write_text("\n".join(report) + "\n")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "comparison")
    parser.add_argument("--models", nargs="+", choices=(*MODEL_NAMES, "OriginalCNN"), default=list(MODEL_NAMES))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        parser.error("Positive epochs/batch-size and image-size >=32 required")
    if len(set(args.models)) != len(args.models): parser.error("Choose distinct models")
    os.environ.setdefault("KERAS_HOME", str(PROJECT_DIR / ".keras"))
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mpl-cache"))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "6")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    classes = discover_classes(args.dataset)
    splits = split_dataset(classes, args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "split_manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "relative_path", "label", "class_name", "image_sha256"])
        root = args.dataset.expanduser().resolve()
        for split, (paths, labels) in splits.items():
            for path, label in zip(paths, labels):
                writer.writerow([split, str(Path(path).relative_to(root)), label, list(classes)[label],
                                 hashlib.sha256(Path(path).read_bytes()).hexdigest()])
    config = {"seed": args.seed, "epochs": args.epochs, "image_size": args.image_size,
              "batch_size": args.batch_size, "class_names": list(classes),
              "split_counts": {name: len(paths) for name, (paths, _) in splits.items()},
              "split_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "encoder_weights": "ImageNet_frozen_except_OriginalCNN_random",
              "head": {"OriginalCNN": "Flatten_Dense8", "transfer": "Dense256_Dropout0.3"},
              "optimizer": "Adam0.001", "early_stopping_patience": 5,
              "class_weights": "inverse_training_frequency"}
    results = []
    # The historical CNN is still available explicitly, after pretrained models.
    ordered_models = [name for name in args.models if name != "OriginalCNN"] + [name for name in args.models if name == "OriginalCNN"]
    for name in ordered_models:
        results.append(run_model(name, classes, splits, config, output))
        generate_report(results, config, output)
    print(f"Comparison complete: {output / 'comparison_report.md'}", flush=True)


if __name__ == "__main__":
    main()
