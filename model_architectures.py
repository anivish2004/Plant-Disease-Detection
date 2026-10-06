"""Serializable ImageNet encoders with their required RGB preprocessing."""

MODEL_NAMES = ("OriginalCNN", "MobileNetV2", "EfficientNetB0", "DenseNet121")


def build_classifier(name, num_classes, image_size=128):
    import tensorflow as tf
    layers = tf.keras.layers
    options = dict(input_shape=(image_size, image_size, 3), include_top=False,
                   weights="imagenet", pooling="avg")
    preprocessing = []
    if name == "OriginalCNN":
        # Preserve the original convolution/pooling/Dense(8) architecture.
        # Class count and shared input resolution are adapted for this experiment.
        model = tf.keras.Sequential([
            layers.Input(shape=(image_size, image_size, 3)), layers.Rescaling(1 / 255),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)), layers.Flatten(),
            layers.Dense(8, activation="relu"), layers.Dense(num_classes, activation="softmax"),
        ], name="OriginalCNN")
        return model, "model_rescaling_1_over_255"
    if name == "MobileNetV2":
        backbone = tf.keras.applications.MobileNetV2(**options)
        preprocessing = [layers.Rescaling(1 / 127.5, offset=-1)]
        normalization = "model_rescaling_1_over_127_5_minus_1"
    elif name == "EfficientNetB0":
        backbone = tf.keras.applications.EfficientNetB0(**options)
        normalization = "model_efficientnet_preprocessing"
    elif name == "DenseNet121":
        backbone = tf.keras.applications.DenseNet121(**options)
        preprocessing = [layers.Rescaling(1 / 255),
                         layers.Normalization(mean=[0.485, 0.456, 0.406],
                                              variance=[0.229**2, 0.224**2, 0.225**2])]
        normalization = "model_imagenet_rgb_normalization"
    else:
        raise ValueError(f"Unknown model: {name}. Choose from {MODEL_NAMES}")
    backbone.trainable = False
    encoder = tf.keras.Sequential([
        layers.Input(shape=(image_size, image_size, 3)), *preprocessing, backbone,
    ], name="image_encoder")
    classifier = tf.keras.Sequential([
        encoder, layers.Dense(256, activation="relu"), layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name=name)
    return classifier, normalization
