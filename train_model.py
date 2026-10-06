"""Train a classifier for every class in PlantVillage raw/color."""
import argparse
import json
import os
from pathlib import Path
from dataset import DEFAULT_DATASET, discover_classes, split_dataset

PROJECT_DIR = Path(__file__).resolve().parent


def make_dataset(paths, labels, image_size, batch_size, training=False, seed=42):
    import tensorflow as tf

    def load_image(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        return tf.image.resize(image, [image_size, image_size]), label

    data = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        data = data.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    return data.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(
        batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes, image_size=128):
    import tensorflow as tf
    layers = tf.keras.layers
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3), include_top=False,
        weights="imagenet", pooling="avg")
    backbone.trainable = False
    encoder = tf.keras.Sequential([
        layers.Input(shape=(image_size, image_size, 3)),
        layers.Rescaling(1.0 / 127.5, offset=-1), backbone,
    ], name="image_encoder")
    model = tf.keras.Sequential([
        encoder, layers.Dense(256, activation="relu"), layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def train(dataset_path=DEFAULT_DATASET, output_dir=PROJECT_DIR, epochs=30,
          batch_size=32, image_size=128, seed=42):
    # Keep downloaded ImageNet weights inside the project, not the home cache.
    os.environ.setdefault("KERAS_HOME", str(PROJECT_DIR / ".keras"))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "6")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    import numpy as np
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    classes = discover_classes(dataset_path)
    splits = split_dataset(classes, seed)
    print(f"Training {len(classes)} classes on {sum(map(len, classes.values()))} images")
    model = build_model(len(classes), image_size)
    model.summary()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Extract features once: only the small classification head needs gradients.
    features = {}
    class FeatureProgress(tf.keras.callbacks.Callback):
        def on_predict_batch_end(self, batch, logs=None):
            if (batch + 1) % 100 == 0:
                print(f"  Extracted {batch + 1} batches", flush=True)

    for name in splits:
        print(f"Extracting {name} features", flush=True)
        ordered_data = make_dataset(*splits[name], image_size, batch_size)
        features[name] = model.layers[0].predict(ordered_data, verbose=2, callbacks=[FeatureProgress()])
    head = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(features["train"].shape[1],)),
        *model.layers[1:],
    ], name="classification_head")
    head.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    def feature_dataset(name, shuffle=False):
        data = tf.data.Dataset.from_tensor_slices(
            (features[name], np.asarray(splits[name][1], dtype="int32")))
        if shuffle:
            data = data.shuffle(len(splits[name][0]), seed=seed)
        return data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_counts = np.bincount(splits["train"][1], minlength=len(classes))
    class_weights = {index: float(len(splits["train"][0]) / (len(classes) * count))
                     for index, count in enumerate(train_counts)}
    history = head.fit(
        feature_dataset("train", shuffle=True),
        validation_data=feature_dataset("validation"), epochs=epochs, verbose=2,
        class_weight=class_weights,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                   restore_best_weights=True),
                   tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2),
                   tf.keras.callbacks.ModelCheckpoint(str(output / "training_head_checkpoint.keras"),
                                                      save_best_only=True),
                   tf.keras.callbacks.CSVLogger(str(output / "training_log.csv")),
                   tf.keras.callbacks.TerminateOnNaN()],
    )
    metrics = head.evaluate(feature_dataset("test"), return_dict=True, verbose=2)
    test_predictions = np.argmax(head.predict(feature_dataset("test"), verbose=0), axis=1)
    actual = np.asarray(splits["test"][1])
    per_class = {name: {"test_images": int(np.sum(actual == index)),
                        "accuracy": float(np.mean(test_predictions[actual == index] == index))}
                 for index, name in enumerate(classes)}
    (output / "test_class_metrics.json").write_text(json.dumps(per_class, indent=2) + "\n")
    # Head layers are shared with the full model, so restored weights export too.
    model.save(output / "plant_disease_model.keras")
    metadata = {"class_names": list(classes), "image_size": image_size, "color_mode": "RGB",
                "normalization": "model_rescaling_1_over_127_5_minus_1", "seed": seed,
                "class_counts": {name: len(paths) for name, paths in classes.items()},
                "split_counts": {name: len(paths) for name, (paths, _) in splits.items()},
                "encoder": "MobileNetV2_ImageNet_frozen",
                "balanced_class_weights": True,
                "test_metrics": {key: float(value) for key, value in metrics.items()}}
    (output / "class_names.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "training_history.json").write_text(json.dumps(history.history, indent=2) + "\n")
    print(f"Saved model and class mapping to {output}; test metrics: {metrics}")
    return model, history, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect", action="store_true", help="List classes without TensorFlow")
    args = parser.parse_args()
    if args.inspect:
        classes = discover_classes(args.dataset)
        for index, (name, images) in enumerate(classes.items()):
            print(f"{index:2d}  {name}: {len(images)}")
        print(f"Total: {len(classes)} classes, {sum(map(len, classes.values()))} images")
        return
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        parser.error("epochs and batch-size must be positive; image-size must be at least 32")
    train(args.dataset, args.output_dir, args.epochs, args.batch_size, args.image_size, args.seed)


if __name__ == "__main__":
    main()
