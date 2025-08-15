"""
Model Training Pipeline Step for YOLOv5 Object Detection.

This module handles the training of YOLOv5 models using preprocessed datasets.
It includes comprehensive device detection, parameter optimization, and training
monitoring with robust error handling.

Key Features:
- Automatic GPU/CPU detection and optimization
- Dynamic batch size adjustment based on available memory
- Comprehensive training parameter validation
- Robust weight file location and management
- Detailed progress tracking and performance metrics

Expected Input:
    - Preprocessed dataset in /data/ directory structure
    - configuration.yaml with training parameters
    - YOLOv5 model weights and code

Output:
    - model.pt: Trained PyTorch model file

Author: Generated for Robot Hackathon Object Detection Pipeline
Version: 2.0 (Enhanced with comprehensive logging and error handling)
"""

from os import environ
from shutil import move
import torch
import logging
import sys
import os
import glob
from datetime import datetime
from pathlib import Path

from yolov5.train import run

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def train_model(
        data_folder='./data', batch_size=0, epochs=0, base_model='yolov5n'):
    """
    Train a YOLOv5 object detection model with comprehensive monitoring and optimization.
    
    This function handles the complete training pipeline including device setup,
    parameter validation, training execution, and output file management. It automatically
    optimizes settings based on available hardware and provides detailed progress tracking.
    
    Args:
        data_folder (str, optional): Path to the directory containing preprocessed
            training data. Defaults to './data'.
        batch_size (int, optional): Training batch size. If 0, uses environment
            variable or optimized default based on hardware. Defaults to 0.
        epochs (int, optional): Number of training epochs. If 0, uses environment
            variable or default. Defaults to 0.
        base_model (str, optional): YOLOv5 model variant to use as starting point.
            Defaults to 'yolov5n'.
    
    Raises:
        FileNotFoundError: If required files (config, data, weights) are missing.
        ValueError: If training parameters are invalid.
        RuntimeError: If training fails or output file cannot be created.
    
    Environment Variables:
        - batch_size: Override default batch size
        - epochs: Override default number of epochs
        - base_model: Override default model variant
    
    Training Optimizations:
        - Automatic GPU/CPU detection and configuration
        - Memory-based batch size optimization
        - Early stopping with patience
        - Cosine learning rate scheduling
        - RAM caching for faster data loading
        - Multi-worker data loading
    
    Output:
        Creates 'model.pt' file in the current directory containing the trained model.
    
    Notes:
        - Uses YOLOv5's built-in training with performance optimizations
        - Automatically handles weight file location and naming
        - Provides comprehensive logging of training progress and metrics
        - Validates all prerequisites before starting training
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    logger.info(f"Start time: {start_time}")
    
    try:
        # Validate prerequisites
        logger.info("🔍 Validating training prerequisites...")
        _validate_training_prerequisites(data_folder)
        
        # Get and validate parameters
        logger.info("⚙️  Processing training parameters...")
        batch_size, epochs, base_model = _get_training_parameters(batch_size, epochs, base_model)
        
        # Setup and validate device
        logger.info("🖥️  Setting up compute device...")
        device = _setup_device(batch_size)
        
        # Adjust batch size based on device capabilities
        if device == '0':  # GPU device
            batch_size = _optimize_batch_size_for_gpu(batch_size)
        
        logger.info(f"📋 Final training configuration:")
        logger.info(f"  🔢 Batch size: {batch_size}")
        logger.info(f"  🔄 Epochs: {epochs}")
        logger.info(f"  🤖 Base model: {base_model}")
        logger.info(f"  💻 Device: {device}")

        # Prepare model weights (YOLOv5 will auto-download if needed)
        logger.info("🏋️  Preparing model weights...")
        weights_file = f'{base_model}.pt'
        logger.info(f"📦 Will use weights: {weights_file} (auto-download if needed)")

        # Start training
        logger.info("🚀 Starting YOLOv5 training...")
        logger.info("📊 Training progress will be shown below:")
        logger.info("-" * 60)
        
        training_start = datetime.now()
        
        try:
            run(
                data='configuration.yaml',
                weights=weights_file,
                epochs=epochs,
                batch_size=batch_size,
                # Performance tuning
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
        except Exception as e:
            error_msg = f"❌ CRITICAL ERROR: YOLOv5 training failed: {str(e)}"
            logger.error(error_msg)
            logger.error("💡 HINT: Check if the training data is properly formatted")
            logger.error("💡 HINT: Verify configuration.yaml has correct paths")
            logger.error("💡 HINT: Check available disk space and memory")
            logger.error("💡 HINT: Reduce batch_size if running out of memory")
            raise RuntimeError(error_msg)
        
        training_end = datetime.now()
        training_duration = training_end - training_start
        logger.info("-" * 60)
        logger.info(f"✅ Training completed successfully in {training_duration}")

        # Find and move trained weights
        logger.info("📦 Locating trained model weights...")
        weights_path = _find_trained_weights()
        
        if weights_path:
            logger.info(f"✅ Found trained weights: {weights_path}")
            _move_weights_to_output(weights_path)
        else:
            error_msg = "❌ CRITICAL ERROR: No trained weights found after training!"
            logger.error(error_msg)
            logger.error("💡 HINT: Training may have failed or been interrupted")
            logger.error("💡 HINT: Check training logs above for errors")
            raise FileNotFoundError(error_msg)

        # Final validation
        if os.path.exists('model.pt'):
            model_size = os.path.getsize('model.pt') / (1024 * 1024)  # MB
            logger.info(f"✅ Output model created: model.pt ({model_size:.1f} MB)")
        else:
            error_msg = "❌ CRITICAL ERROR: Final model.pt was not created!"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Log final statistics
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total time: {total_duration}")
        logger.info(f"🏋️  Training time: {training_duration}")
        logger.info(f"🎯 Epochs completed: {epochs}")
        logger.info(f"📦 Output: model.pt ({model_size:.1f} MB)")
        logger.info(f"🏁 End time: {end_time}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ MODEL TRAINING FAILED!")
        logger.error("=" * 60)
        logger.error(f"💥 Error: {str(e)}")
        logger.error(f"⏱️  Failed after: {datetime.now() - start_time}")
        logger.error("=" * 60)
        raise


def _validate_training_prerequisites(data_folder):
    """
    Validate that all required files and directories exist for training.
    
    This function performs comprehensive checks to ensure the training environment
    is properly set up before attempting to start training.
    
    Args:
        data_folder (str): Path to the data directory to validate.
    
    Raises:
        FileNotFoundError: If any required files or directories are missing.
    
    Validation Checks:
        - configuration.yaml exists and is accessible
        - yolov5 directory exists (contains training code)
        - Required training data directories exist and contain files:
            * data_folder/images/train/
            * data_folder/images/val/
            * data_folder/labels/train/
            * data_folder/labels/val/
    
    Notes:
        - Logs detailed information about each validation check
        - Provides specific error messages and hints for missing components
        - Counts files in each directory to ensure data is present
    """
    # Check configuration file
    if not os.path.exists('configuration.yaml'):
        error_msg = "❌ CRITICAL ERROR: configuration.yaml not found!"
        logger.error(error_msg)
        logger.error("💡 HINT: Ensure configuration.yaml is included in pipeline dependencies")
        raise FileNotFoundError(error_msg)
    
    logger.info("✅ Configuration file found")
    
    # Check YOLOv5 directory
    if not os.path.exists('yolov5'):
        error_msg = "❌ CRITICAL ERROR: yolov5 directory not found!"
        logger.error(error_msg)
        logger.error("💡 HINT: Ensure yolov5/* is included in pipeline dependencies")
        raise FileNotFoundError(error_msg)
    
    logger.info("✅ YOLOv5 directory found")
    
    # Check training data
    required_paths = [
        f'{data_folder}/images/train',
        f'{data_folder}/images/val',
        f'{data_folder}/labels/train',
        f'{data_folder}/labels/val'
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            error_msg = f"❌ CRITICAL ERROR: Required training data path '{path}' not found!"
            logger.error(error_msg)
            logger.error("💡 HINT: Ensure the preprocessing step completed successfully")
            logger.error("💡 HINT: Check if the data_folder parameter is correct")
            raise FileNotFoundError(error_msg)
        
        # Check if directory has files
        files = os.listdir(path)
        if not files:
            error_msg = f"❌ CRITICAL ERROR: Training data directory '{path}' is empty!"
            logger.error(error_msg)
            logger.error("💡 HINT: Ensure the preprocessing step populated the directories")
            raise FileNotFoundError(error_msg)
        
        logger.info(f"✅ Training data found: {path} ({len(files)} files)")


def _get_training_parameters(batch_size, epochs, base_model):
    """
    Process and validate training parameters from multiple sources.
    
    This function handles parameter resolution from function arguments,
    environment variables, and defaults, with comprehensive validation.
    
    Args:
        batch_size (int): Batch size parameter (0 means use environment/default).
        epochs (int): Number of epochs (0 means use environment/default).
        base_model (str): YOLOv5 model variant name.
    
    Returns:
        tuple: A tuple containing (batch_size, epochs, base_model) with
            resolved and validated values.
    
    Raises:
        ValueError: If any parameter is invalid or out of acceptable range.
    
    Parameter Resolution Order:
        1. Function parameter (if non-zero/non-empty)
        2. Environment variable
        3. Hard-coded default
    
    Valid Models:
        - yolov5n: Nano (smallest, fastest)
        - yolov5s: Small
        - yolov5m: Medium
        - yolov5l: Large
        - yolov5x: Extra Large (largest, most accurate)
    
    Notes:
        - Logs parameter sources for debugging
        - Validates numeric parameters are positive
        - Ensures model variant is supported
    """
    # Get parameters with optimized defaults
    batch_size = batch_size or int(environ.get('batch_size', 32))
    epochs = epochs or int(environ.get('epochs', 5))
    base_model = environ.get('base_model', base_model or 'yolov5n')
    
    logger.info(f"📋 Parameter sources:")
    logger.info(f"  🔢 Batch size: {batch_size} (from: {'parameter' if batch_size else 'environment/default'})")
    logger.info(f"  🔄 Epochs: {epochs} (from: {'parameter' if epochs else 'environment/default'})")
    logger.info(f"  🤖 Base model: {base_model} (from: environment/parameter/default)")
    
    # Validate parameters
    if batch_size <= 0:
        error_msg = f"❌ CRITICAL ERROR: Invalid batch_size: {batch_size}"
        logger.error(error_msg)
        logger.error("💡 HINT: batch_size must be a positive integer")
        raise ValueError(error_msg)
    
    if epochs <= 0:
        error_msg = f"❌ CRITICAL ERROR: Invalid epochs: {epochs}"
        logger.error(error_msg)
        logger.error("💡 HINT: epochs must be a positive integer")
        raise ValueError(error_msg)
    
    valid_models = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
    if base_model not in valid_models:
        error_msg = f"❌ CRITICAL ERROR: Invalid base_model: {base_model}"
        logger.error(error_msg)
        logger.error(f"💡 HINT: Valid models are: {valid_models}")
        raise ValueError(error_msg)
    
    return batch_size, epochs, base_model


def _setup_device(batch_size):
    """
    Detect and configure the optimal compute device for training.
    
    This function detects available hardware (GPU/CPU) and configures
    the training device accordingly, with warnings and recommendations
    for optimal performance.
    
    Args:
        batch_size (int): Current batch size for memory considerations.
    
    Returns:
        str: Device identifier for YOLOv5 training ('0' for GPU, 'cpu' for CPU).
    
    Notes:
        - Detects CUDA availability and GPU count
        - Logs detailed GPU information (name, memory)
        - Provides performance warnings for CPU usage
        - Suggests batch size adjustments for CPU training
        - Returns YOLOv5-compatible device format
    
    GPU Detection:
        - Checks torch.cuda.is_available()
        - Validates actual GPU devices are present
        - Logs GPU properties for each available device
    
    Performance Recommendations:
        - GPU training is strongly preferred
        - CPU training warnings for large batch sizes
        - Memory considerations for different hardware
    """
    # Check if GPU is available AND has actual devices
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    
    logger.info(f"💻 CUDA availability: {cuda_available}")
    logger.info(f"🖥️  GPU device count: {device_count}")
    
    if cuda_available and device_count > 0:
        device = '0'  # YOLOv5 expects '0' for first GPU, not 'cuda'
        logger.info(f"✅ Using GPU device: {device}")
        
        # Log GPU information
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            logger.info(f"  🎮 GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    else:
        device = 'cpu'
        logger.info("⚠️  GPU not available, using CPU")
        logger.info("💡 HINT: Training on CPU will be significantly slower")
        logger.info("💡 HINT: Consider using GPU-enabled pipeline for faster training")
        
        # Adjust batch size for CPU
        if batch_size > 16:
            logger.warning(f"⚠️  Large batch size ({batch_size}) on CPU may cause memory issues")
            logger.warning("💡 HINT: Consider reducing batch_size for CPU training")
    
    return device


def _optimize_batch_size_for_gpu(batch_size):
    """
    Automatically optimize batch size based on available GPU memory.
    
    This function analyzes GPU memory capacity and adjusts the batch size
    to maximize training efficiency while avoiding out-of-memory errors.
    
    Args:
        batch_size (int): Initial batch size to potentially optimize.
    
    Returns:
        int: Optimized batch size based on GPU memory capacity.
    
    Optimization Logic:
        - GPU > 10GB: Increase default batch size to 64
        - GPU < 6GB: Decrease default batch size to 16
        - Custom batch sizes: Keep user-specified values
    
    Notes:
        - Only optimizes default batch size (32)
        - Preserves user-specified batch sizes
        - Logs optimization decisions and rationale
        - Handles exceptions gracefully
        - Returns original value if optimization fails
    
    Memory Considerations:
        - Larger batch sizes improve training stability
        - Smaller batch sizes reduce memory usage
        - YOLOv5 typically works well with batch sizes 16-64
    """
    try:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🎮 GPU memory: {gpu_memory:.1f} GB")
        
        original_batch_size = batch_size
        
        # Adjust batch size based on GPU memory if using default
        if batch_size == 32:  # Only adjust if using default
            if gpu_memory > 10:
                batch_size = 64  # Increase batch size for larger GPUs
                logger.info(f"🚀 Optimized batch size for large GPU: {original_batch_size} → {batch_size}")
            elif gpu_memory < 6:
                batch_size = 16  # Decrease batch size for smaller GPUs
                logger.info(f"⚡ Optimized batch size for small GPU: {original_batch_size} → {batch_size}")
        else:
            logger.info(f"📌 Using specified batch size: {batch_size}")
        
        return batch_size
        
    except Exception as e:
        logger.warning(f"⚠️  Could not optimize batch size: {e}")
        return batch_size


def _find_trained_weights():
    """
    Locate the trained model weights file after training completion.
    
    This function searches multiple possible locations where YOLOv5 might
    save the best model weights, handling different YOLOv5 configurations
    and directory structures.
    
    Returns:
        str or None: Path to the best weights file if found, None otherwise.
    
    Search Patterns:
        - yolov5/runs/train/exp/weights/best.pt
        - yolov5/runs/train/exp*/weights/best.pt (multiple experiments)
        - runs/train/exp/weights/best.pt
        - runs/train/exp*/weights/best.pt
    
    Fallback Strategy:
        - If best.pt not found, searches for any .pt files
        - Selects most recently modified file
        - Provides detailed debugging information
    
    Notes:
        - Searches in order of preference (best.pt first)
        - Logs all search attempts for debugging
        - Lists all available files if weights not found
        - Handles multiple experiment directories
        - Returns None if no weights found anywhere
    """
    logger.info("🔍 Searching for trained weights...")
    
    # Try multiple possible locations for the weights
    possible_patterns = [
        'yolov5/runs/train/exp/weights/best.pt',
        'yolov5/runs/train/exp*/weights/best.pt',
        'runs/train/exp/weights/best.pt',
        'runs/train/exp*/weights/best.pt'
    ]
    
    weights_path = None
    for pattern in possible_patterns:
        logger.info(f"  🔍 Checking pattern: {pattern}")
        matches = glob.glob(pattern)
        if matches:
            # Sort by modification time, get the most recent
            weights_path = max(matches, key=os.path.getmtime)
            logger.info(f"  ✅ Found weights: {weights_path}")
            break
        else:
            logger.info(f"  ❌ No matches for: {pattern}")
    
    if not weights_path:
        # List available files for debugging
        logger.error("❌ No weights found in expected locations!")
        logger.error("🔍 Debugging - listing available files...")
        
        for search_dir in ['yolov5/runs/train/', 'runs/train/']:
            if os.path.exists(search_dir):
                logger.error(f"📁 Contents of {search_dir}:")
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        logger.error(f"  📄 {full_path}")
            else:
                logger.error(f"📁 Directory does not exist: {search_dir}")
        
        # Try to find any .pt files as fallback
        logger.error("🔍 Searching for any .pt files...")
        pt_files = glob.glob('**/runs/train/**/*.pt', recursive=True)
        if pt_files:
            latest_pt = max(pt_files, key=os.path.getmtime)
            logger.warning(f"⚠️  Found fallback weights: {latest_pt}")
            return latest_pt
        else:
            logger.error("❌ No .pt files found anywhere!")
    
    return weights_path


def _move_weights_to_output(weights_path):
    """
    Move trained weights to the standard output location with backup handling.
    
    This function safely moves the trained weights file to 'model.pt',
    handling existing files and providing verification of the operation.
    
    Args:
        weights_path (str): Path to the source weights file to move.
    
    Raises:
        FileNotFoundError: If the move operation fails to create the output file.
        RuntimeError: If file operations fail due to permissions or disk space.
    
    Backup Strategy:
        - Existing model.pt files are backed up with timestamp
        - Backup format: model_backup_YYYYMMDD_HHMMSS.pt
        - Original file is preserved before overwriting
    
    Verification:
        - Confirms successful move operation
        - Validates output file exists and is accessible
        - Reports file sizes and operation success
    
    Notes:
        - Uses shutil.move() for atomic file operations
        - Provides detailed logging of all file operations
        - Handles edge cases like missing source files
        - Ensures pipeline output consistency
    """
    try:
        if os.path.exists('model.pt'):
            logger.info("⚠️  Existing model.pt found, backing up...")
            backup_name = f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            move('model.pt', backup_name)
            logger.info(f"📦 Backed up to: {backup_name}")
        
        move(weights_path, 'model.pt')
        logger.info(f"✅ Successfully moved weights: {weights_path} → model.pt")
        
        # Verify the move was successful
        if not os.path.exists('model.pt'):
            error_msg = "❌ CRITICAL ERROR: model.pt was not created after move operation!"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
    except Exception as e:
        error_msg = f"❌ CRITICAL ERROR: Failed to move weights file: {str(e)}"
        logger.error(error_msg)
        logger.error(f"💡 HINT: Source: {weights_path}")
        logger.error("💡 HINT: Check file permissions and disk space")
        raise RuntimeError(error_msg)


if __name__ == '__main__':
    """
    Main execution block for standalone script usage.
    
    When run directly, this script trains a model using data from the '/data'
    directory, which is the standard mount point in Kubeflow pipeline environments.
    Training parameters are taken from environment variables or defaults.
    """
    train_model(data_folder='/data')
