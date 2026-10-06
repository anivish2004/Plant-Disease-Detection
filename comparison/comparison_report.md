# Four-model comparison

Preferred model by validation macro F1: **DenseNet121**.

All models use the same 38-class PlantVillage raw/color dataset and stratified image splits.
RGB images are resized to 128 × 128 for all models.
OriginalCNN preserves the repository Conv32/Pool3/Conv16/Pool2/Flatten/Dense8 structure and trains from scratch, with 38 output classes. The prior input resolution was 256; this experiment uses the common resolution above.
The other three use frozen ImageNet encoders and a Dense(256)/Dropout(0.3) classification head.
Maximum 30 epochs, Adam(0.001), inverse-frequency class weights, and identical early-stopping policies.

| Model | Test accuracy | Test macro F1 | Validation macro F1 | Training minutes | Batch-32 ms/image | Size MB |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 96.86% | 96.06% | 96.10% | 6.21 | 4.76 | 10.98 |
| EfficientNetB0 | 97.44% | 96.72% | 96.15% | 8.00 | 7.12 | 18.40 |
| DenseNet121 | 97.59% | 96.88% | 96.70% | 18.36 | 20.32 | 30.77 |
| OriginalCNN | 51.27% | 44.21% | 44.10% | 33.98 | 0.55 | 0.79 |

![Comparison](comparison_chart.png)

Training timings are from this CPU and include train/validation feature extraction plus fitting, excluding test extraction, weight downloads, and model construction. Inference timing excludes image decoding and resizing.
This is one seeded comparison of a CNN trained from scratch versus frozen pretrained encoders at a common resolution, not a full fine-tuning or native-resolution benchmark.
The test split contains 8,129 images; scores describe held-out PlantVillage images, not field performance.
Model selection uses validation macro F1. The current deployed classifier is preserved; choose a comparison model in the app to try it.
Saved split_manifest.csv fixes image membership and class order, and includes each image SHA256 for cache integrity.
