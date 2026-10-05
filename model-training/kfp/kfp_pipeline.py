"""
KFP v2 Pipeline – YOLOv5 Object Detection Model Training.

Replaces the Elyra pipeline (model-training-cpu.pipeline) with a
pure-Python KFP SDK pipeline that can be deployed to Data Science
Pipelines on OpenShift AI.

Each component is **fully self-contained** — all logic is inline,
so there is no need to copy scripts or code to the PVC.  The PVC
is used exclusively for training data and intermediate model files.

Pipeline steps
--------------
1. preprocess_data   – organise images into train / val / test splits
2. train_model       – train a YOLOv5 model on CPU
3. convert_model     – export the trained .pt model to ONNX
4. upload_model      – upload the ONNX model to S3 storage

Usage
-----
Deploy from the companion notebook or compile on the CLI::

    python kfp_pipeline.py          # produces pipeline.yaml
"""

from kfp import dsl
from kfp.kubernetes import mount_pvc, use_secret_as_env

# ── Shared configuration ──────────────────────────────────────────

RUNTIME_IMAGE = (
    'quay.io/cloud-native-robotz-hackathon/pipeline-runtime-image:v2.25'
)
DATA_PVC_NAME = 'object-detection-training-pvc'
DATA_PVC_MOUNT = '/data'
SHM_PVC_NAME = 'shared-memory-pvc'
S3_SECRET_NAME = 'workbench-bucket-ai-connection'


# ── Pipeline components ───────────────────────────────────────────

@dsl.component(base_image=RUNTIME_IMAGE)
def preprocess_data(class_names: str):
    """Split custom training images into train / val / test sets.

    Expects images on the PVC at:
        /data/custom_training_images/<class>/images/*.jpg
        /data/custom_training_images/<class>/labels/*.txt

    Produces:
        /data/images/{train,val,test}/
        /data/labels/{train,val,test}/
    """
    import os
    import random
    from glob import glob
    from math import floor
    from shutil import copy

    data_folder = '/data'
    download_folder = os.path.join(data_folder, 'custom_training_images')

    # Pre-cache YOLOv5 v7.0 repo on the PVC so train/convert steps
    # don't need to download from GitHub (avoids rate-limits and SPOF).
    yolov5_pvc = os.path.join(data_folder, 'yolov5')
    if not os.path.isdir(yolov5_pvc):
        import urllib.request
        import zipfile

        url = 'https://github.com/ultralytics/yolov5/archive/refs/tags/v7.0.zip'
        zip_path = os.path.join(data_folder, 'yolov5.zip')
        print('Pre-caching ultralytics YOLOv5 v7.0 on PVC …')
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_folder)
        os.rename(os.path.join(data_folder, 'yolov5-7.0'), yolov5_pvc)
        os.remove(zip_path)
        print('YOLOv5 v7.0 cached at /data/yolov5/')
    else:
        print('YOLOv5 v7.0 already cached on PVC.')

    classes = [c.strip() for c in class_names.split(',')]
    folder_names = [c.lower() for c in classes]

    # Create target directories
    for kind in ['images', 'labels']:
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(data_folder, kind, split), exist_ok=True)

    random.seed(42)
    train_ratio, val_ratio = 0.75, 0.125

    for folder_name in folder_names:
        image_dir = os.path.join(download_folder, folder_name, 'images')
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(
                f'Image directory not found: {image_dir}'
            )

        filenames = sorted(
            os.path.basename(p)
            for p in glob(os.path.join(image_dir, '*.jpg'))
        )
        if not filenames:
            raise FileNotFoundError(f'No .jpg images in {image_dir}')

        random.shuffle(filenames)
        train_size = floor(train_ratio * len(filenames))
        val_size = floor(val_ratio * len(filenames))

        for i, name in enumerate(filenames):
            if i < train_size:
                split = 'train'
            elif i < train_size + val_size:
                split = 'val'
            else:
                split = 'test'

            copy(
                os.path.join(download_folder, folder_name, 'images', name),
                os.path.join(data_folder, 'images', split),
            )
            label = name.replace('.jpg', '.txt')
            label_src = os.path.join(
                download_folder, folder_name, 'labels', label,
            )
            if os.path.exists(label_src):
                copy(label_src, os.path.join(data_folder, 'labels', split))

        test_size = len(filenames) - train_size - val_size
        print(
            f'{folder_name}: {len(filenames)} images '
            f'(train={train_size}, val={val_size}, test={test_size})'
        )

    print('Preprocessing complete.')


@dsl.component(base_image=RUNTIME_IMAGE)
def train_model(
    batch_size: int, epochs: int, base_model: str, class_names: str,
):
    """Train a YOLOv5 object-detection model on CPU.

    Uses the original ultralytics YOLOv5 code (same as the Elyra
    pipeline's bundled yolov5/ directory) — NOT the fcakyon pip
    package which has different convergence behavior.

    Reads preprocessed data from the PVC at /data/images/ and
    writes the trained model to /data/model.pt.
    """
    import glob
    import os
    import sys
    from shutil import move

    # Set thread counts before importing PyTorch to avoid
    # oversubscription when multiple pods share a node.
    os.environ['OMP_NUM_THREADS'] = '32'
    os.environ['MKL_NUM_THREADS'] = '32'

    workdir = '/tmp/training'
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    # Use YOLOv5 v7.0 pre-cached on the PVC by the preprocess step.
    yolov5_dir = '/data/yolov5'
    if not os.path.isdir(yolov5_dir):
        raise FileNotFoundError(
            'YOLOv5 not found at /data/yolov5 — '
            'preprocess step should have cached it'
        )
    print('Using PVC-cached YOLOv5 v7.0')

    sys.path.insert(0, yolov5_dir)

    import torch
    import yaml

    # Generate configuration.yaml
    classes = [c.strip() for c in class_names.split(',')]
    config = {
        'train': '/data/images/train',
        'val': '/data/images/val',
        'test': '/data/images/test',
        'nc': len(classes),
        'names': classes,
    }
    config_path = os.path.join(workdir, 'configuration.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f'Configuration written to {config_path}')

    # Detect device
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = '0'
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'
        print('Using CPU')

    # Run YOLOv5 training (ultralytics repo version)
    from train import run as train_run

    train_run(
        data=config_path,
        weights=f'{base_model}.pt',
        epochs=epochs,
        batch_size=batch_size,
        freeze=[10],
        cache='ram',
        device=device,
        workers=8,
        project=os.path.join(workdir, 'runs', 'train'),
        exist_ok=True,
        save_period=5,
        patience=50,
        single_cls=True,
        rect=True,
        cos_lr=True,
        close_mosaic=10,
    )

    # Locate and move trained weights to /data/model.pt
    search_patterns = [
        os.path.join(workdir, 'runs/train/exp/weights/best.pt'),
        os.path.join(workdir, 'runs/train/exp*/weights/best.pt'),
    ]
    weights_path = None
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            weights_path = max(matches, key=os.path.getmtime)
            break

    if not weights_path:
        pt_files = glob.glob(
            os.path.join(workdir, 'runs/train/**/*.pt'), recursive=True,
        )
        if pt_files:
            weights_path = max(pt_files, key=os.path.getmtime)

    if not weights_path:
        raise FileNotFoundError('No trained weights found after training!')

    output_path = '/data/model.pt'
    move(weights_path, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'Model saved to {output_path} ({size_mb:.1f} MB)')


@dsl.component(base_image=RUNTIME_IMAGE)
def convert_model():
    """Convert the trained PyTorch model (.pt) to ONNX.

    Reads /data/model.pt, writes /data/model.onnx.
    Uses the ultralytics YOLOv5 repo code (same as train step).
    """
    import os
    import sys

    workdir = '/tmp/convert'
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    # Use YOLOv5 v7.0 pre-cached on the PVC by the preprocess step.
    yolov5_dir = '/data/yolov5'
    if not os.path.isdir(yolov5_dir):
        raise FileNotFoundError(
            'YOLOv5 not found at /data/yolov5 — '
            'preprocess step should have cached it'
        )
    print('Using PVC-cached YOLOv5 v7.0')

    sys.path.insert(0, yolov5_dir)

    model_pt = '/data/model.pt'
    if not os.path.exists(model_pt):
        raise FileNotFoundError(f'{model_pt} not found')

    from export import run as export_run

    export_run(
        weights=model_pt,
        include=['onnx'],
        imgsz=(640, 640),
        opset=13,
        device='cpu',
        verbose=True,
    )

    model_onnx = '/data/model.onnx'
    if not os.path.exists(model_onnx):
        raise FileNotFoundError(
            f'Conversion did not produce {model_onnx}'
        )

    size_mb = os.path.getsize(model_onnx) / (1024 * 1024)
    print(f'ONNX model saved to {model_onnx} ({size_mb:.1f} MB)')


@dsl.component(base_image=RUNTIME_IMAGE)
def upload_model(model_object_prefix: str):
    """Upload the ONNX model to S3-compatible storage.

    Reads /data/model.onnx and uploads it to the bucket configured
    via the UPLOAD_AWS_* environment variables (injected from a
    Kubernetes secret).

    Storage layout (OVMS / KServe):
        models/{prefix}/{timestamp}/model.onnx   (versioned)
        models/{prefix}/1/model.onnx              (latest / serving)
    """
    import os
    from datetime import datetime

    from boto3 import client as boto3_client

    model_path = '/data/model.onnx'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'{model_path} not found')

    s3_endpoint = os.environ['UPLOAD_AWS_S3_ENDPOINT']
    s3_bucket = os.environ['UPLOAD_AWS_S3_BUCKET']

    s3 = boto3_client(
        's3',
        endpoint_url=s3_endpoint,
        aws_access_key_id=os.environ['UPLOAD_AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['UPLOAD_AWS_SECRET_ACCESS_KEY'],
    )

    version = datetime.now().strftime('%y%m%d%H%M')
    keys = [
        f'models/{model_object_prefix}/{version}/model.onnx',
        f'models/{model_object_prefix}/1/model.onnx',
    ]

    for key in keys:
        s3.upload_file(model_path, s3_bucket, key)
        print(f'Uploaded: s3://{s3_bucket}/{key}')

    serving_path = f'models/{model_object_prefix}'
    print(f'OpenShift AI / OVMS model path: {serving_path}')
    print('Upload complete.')


# ── Pipeline definition ───────────────────────────────────────────

@dsl.pipeline(name='object-detection-model-training')
def model_training_pipeline(
    batch_size: int = 16,
    epochs: int = 300,
    base_model: str = 'yolov5n',
    model_object_prefix: str = 'model',
    class_names: str = 'Fedora',
):
    """End-to-end YOLOv5 object-detection training pipeline (CPU)."""

    # Step 1 — Preprocessing ────────────────────────────────────
    preprocess_task = preprocess_data(class_names=class_names)
    mount_pvc(
        preprocess_task,
        pvc_name=DATA_PVC_NAME,
        mount_path=DATA_PVC_MOUNT,
    )
    preprocess_task.set_caching_options(False)
    preprocess_task.set_cpu_request('2').set_memory_request('8G')

    # Step 2 — Model training ───────────────────────────────────
    train_task = train_model(
        batch_size=batch_size,
        epochs=epochs,
        base_model=base_model,
        class_names=class_names,
    )
    mount_pvc(
        train_task,
        pvc_name=DATA_PVC_NAME,
        mount_path=DATA_PVC_MOUNT,
    )
    # Mount the shared-memory PVC at /dev/shm (same as Elyra pipeline).
    # Using a real PVC avoids the emptyDir Memory penalty where tmpfs
    # usage counts against the container memory cgroup.
    mount_pvc(
        train_task,
        pvc_name=SHM_PVC_NAME,
        mount_path='/dev/shm',
    )
    train_task.after(preprocess_task)
    train_task.set_caching_options(False)
    # Request 16 CPUs to force the scheduler to place at most one
    # training pod per 32-CPU node, eliminating CPU burst contention
    # when 9 pipelines run in parallel.  No limits — pod can still
    # burst to all 32 node CPUs.
    train_task.set_cpu_request('16').set_memory_request('8G')

    # Step 3 — ONNX conversion ─────────────────────────────────
    convert_task = convert_model()
    mount_pvc(
        convert_task,
        pvc_name=DATA_PVC_NAME,
        mount_path=DATA_PVC_MOUNT,
    )
    convert_task.after(train_task)
    convert_task.set_caching_options(False)
    convert_task.set_cpu_request('2').set_memory_request('4G')

    # Step 4 — Upload to S3 ────────────────────────────────────
    upload_task = upload_model(model_object_prefix=model_object_prefix)
    mount_pvc(
        upload_task,
        pvc_name=DATA_PVC_NAME,
        mount_path=DATA_PVC_MOUNT,
    )
    use_secret_as_env(
        upload_task,
        secret_name=S3_SECRET_NAME,
        secret_key_to_env={
            'AWS_SECRET_ACCESS_KEY': 'UPLOAD_AWS_SECRET_ACCESS_KEY',
            'AWS_ACCESS_KEY_ID': 'UPLOAD_AWS_ACCESS_KEY_ID',
            'AWS_S3_BUCKET': 'UPLOAD_AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'UPLOAD_AWS_S3_ENDPOINT',
        },
    )
    upload_task.after(convert_task)
    upload_task.set_caching_options(False)
    upload_task.set_cpu_request('1').set_memory_request('2G')


# ── CLI entry-point: compile to YAML ──────────────────────────────

if __name__ == '__main__':
    from kfp.compiler import Compiler

    out = 'pipeline.yaml'
    Compiler().compile(model_training_pipeline, package_path=out)
    print(f'Pipeline compiled to {out}')
