from os import environ
from shutil import move
import torch

from yolov5.train import run


def train_model(
        data_folder='./data', batch_size=0, epochs=0, base_model='yolov5n'):
    print('training model with optimized settings')

    # Get parameters with optimized defaults
    batch_size = batch_size or int(environ.get('batch_size', 32))
    epochs = epochs or int(environ.get('epochs', 5))
    base_model = environ.get('base_model', 'yolov5n')
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Optimize batch size for GPU memory
    if device == 'cuda':
        # Get GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'GPU memory: {gpu_memory:.1f} GB')
        
        # Adjust batch size based on GPU memory if not specified
        if batch_size == 32 and gpu_memory > 10:
            batch_size = 64  # Increase batch size for larger GPUs
        elif batch_size == 32 and gpu_memory < 6:
            batch_size = 16  # Decrease batch size for smaller GPUs

    print(f'Training with: batch_size={batch_size}, epochs={epochs}, model={base_model}')

    run(
        data='configuration.yaml',
        weights=f'{base_model}.pt',
        epochs=epochs,
        batch_size=batch_size,
        freeze=[10],  # Freeze first 10 layers for faster training
        cache='ram',  # Use RAM caching for faster data loading
        device=device,
        workers=8,  # Use more workers for data loading
        project='runs/train',
        exist_ok=True,
        save_period=5,  # Save checkpoint every 5 epochs
        patience=10,  # Early stopping patience
        # Optimizations for faster training
        single_cls=True,  # Single class optimization
        rect=True,  # Rectangular training for efficiency
        cos_lr=True,  # Cosine LR scheduler
        close_mosaic=10,  # Close mosaic augmentation for last 10 epochs
    )

    # Find the weights file more robustly
    import glob
    import os
    
    # Try multiple possible locations for the weights
    possible_paths = [
        'yolov5/runs/train/exp/weights/best.pt',
        'yolov5/runs/train/exp*/weights/best.pt',
        'runs/train/exp/weights/best.pt',
        'runs/train/exp*/weights/best.pt'
    ]
    
    weights_path = None
    for pattern in possible_paths:
        matches = glob.glob(pattern)
        if matches:
            # Sort by modification time, get the most recent
            weights_path = max(matches, key=os.path.getmtime)
            print(f'Found weights at: {weights_path}')
            break
    
    if weights_path and os.path.exists(weights_path):
        move(weights_path, 'model.pt')
        print(f'Successfully moved weights from {weights_path} to model.pt')
    else:
        # List available files for debugging
        print('ERROR: Could not find weights file!')
        print('Available files in yolov5/runs/train/:')
        for root, dirs, files in os.walk('yolov5/runs/train/'):
            for file in files:
                print(f'  {os.path.join(root, file)}')
        
        # Try to find any .pt files as fallback
        pt_files = glob.glob('yolov5/runs/train/**/*.pt', recursive=True)
        if pt_files:
            latest_pt = max(pt_files, key=os.path.getmtime)
            print(f'Found fallback weights: {latest_pt}')
            move(latest_pt, 'model.pt')
        else:
            raise FileNotFoundError('No weights file found after training!')

    print('model training done')


if __name__ == '__main__':
    train_model(data_folder='/data')
