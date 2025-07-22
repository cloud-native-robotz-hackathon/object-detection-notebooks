# Object Detection Notebooks

YOLOv5-based object detection training pipeline for OpenShift/Kubernetes with Kubeflow Pipelines.

## Features

- YOLOv5 object detection training
- CPU and GPU support with automatic fallback
- Kubeflow pipeline automation
- ONNX model conversion
- OpenShift/Kubernetes deployment ready

## Project Structure

```
object-detection-notebooks/
├── model-training/
│   ├── configuration.yaml             # Dataset configuration
│   ├── data_ingestion.py              # Data preparation
│   ├── preprocessing.py               # Data preprocessing
│   ├── model_training.py              # Main training script
│   ├── model_conversion.py            # ONNX conversion
│   ├── model_upload.py                # Model upload
│   ├── model-training-cpu.pipeline    # CPU pipeline
│   ├── model-training-gpu.pipeline    # GPU pipeline
│   └── yolov5/                        # YOLOv5 framework
├── sample-images/                     # Sample images
├── object_detection.py               # Inference script
└── requirements.txt                   # Dependencies
```

## Quick Start

### Prerequisites
- OpenShift/Kubernetes cluster with Data Science Pipelines
- S3-compatible storage

### Setup
1. Place training images in `/data/images/train` and `/data/images/val`
2. Update `model-training/configuration.yaml` with your classes:
   ```yaml
   nc: 1
   names: ['YourObjectClass']
   ```
3. Run the appropriate pipeline:
   - `model-training-gpu.pipeline` for GPU training (with CPU fallback)
   - `model-training-cpu.pipeline` for CPU-only training

## Configuration

### Training Parameters
- `batch_size`: Training batch size (default: 64)
- `epochs`: Number of epochs (default: 8)
- `base_model`: YOLOv5 variant (default: 'yolov5n')
- `sample_count`: Number of samples (default: 5000)

## Pipeline Stages
1. **Data Ingestion**: Load and prepare training data
2. **Preprocessing**: Image processing and augmentation
3. **Model Training**: YOLOv5 training with device auto-detection
4. **Model Conversion**: Convert to ONNX format
5. **Model Upload**: Upload to S3 storage

## Troubleshooting

### Common Issues
- **GPU Detection Error**: Code automatically falls back to CPU
- **Memory Issues**: Reduce `batch_size` or use CPU pipeline
- **Import Errors**: Dependencies auto-install during pipeline run

### Debug Device Selection
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
```

## Usage

### Training
Run the pipeline through Data Science Pipelines UI or CLI.

### Inference
```python
from object_detection import detect_objects
results = detect_objects('image.jpg', 'model.pt')
```

## Support
- Check pipeline logs for detailed errors
- Review troubleshooting section above
- Create repository issues for bugs