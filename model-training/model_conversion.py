"""
Model Conversion Pipeline Step for YOLOv5 Object Detection.

This module handles the conversion of trained PyTorch YOLOv5 models to ONNX format
for deployment and inference optimization. The conversion process includes validation,
performance monitoring, and comprehensive error handling.

Key Features:
- PyTorch to ONNX model conversion
- Input/output validation with size verification
- Conversion performance monitoring
- Comprehensive error handling and debugging
- Support for standard YOLOv5 model formats

Expected Input:
    - model.pt: Trained PyTorch model file from training step

Output:
    - model.onnx: Converted ONNX model file ready for deployment

Conversion Specifications:
    - Target format: ONNX
    - Input image size: 640x640 pixels
    - ONNX opset version: 13 (for broad compatibility)
    - Device: CPU (for maximum compatibility)

Author: Generated for Robot Hackathon Object Detection Pipeline
Version: 2.0 (Enhanced with comprehensive logging and error handling)
"""

from yolov5.export import run
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def convert_model(model_file_path='model.pt'):
    """
    Convert a trained YOLOv5 PyTorch model to ONNX format with comprehensive validation.
    
    This function handles the complete model conversion pipeline including input validation,
    ONNX conversion execution, output verification, and performance analysis. The conversion
    uses optimized settings for deployment compatibility.
    
    Args:
        model_file_path (str, optional): Path to the input PyTorch model file.
            Defaults to 'model.pt'.
    
    Raises:
        FileNotFoundError: If the input model file doesn't exist or output file
            is not created.
        ImportError: If YOLOv5 export module cannot be imported.
        RuntimeError: If the ONNX conversion process fails.
    
    Conversion Process:
        1. Validate input model file exists and check size
        2. Verify YOLOv5 export module availability
        3. Execute ONNX conversion with optimized parameters
        4. Validate output file creation and integrity
        5. Perform size comparison and analysis
    
    Conversion Parameters:
        - Format: ONNX (Open Neural Network Exchange)
        - Input size: 640x640 pixels (standard for YOLOv5)
        - ONNX opset: 13 (for broad framework compatibility)
        - Device: CPU (ensures maximum deployment compatibility)
        - Verbose: True (detailed conversion logging)
    
    Output Validation:
        - Confirms model.onnx file creation
        - Verifies reasonable file size
        - Compares input/output file sizes
        - Detects potential conversion issues
    
    Performance Metrics:
        - Conversion duration tracking
        - File size comparison analysis
        - Conversion speed monitoring
    
    Notes:
        - Uses CPU for conversion to avoid GPU memory issues
        - Provides detailed debugging information on failures
        - Optimized for deployment rather than training
        - Compatible with standard ONNX runtime environments
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL CONVERSION")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    logger.info(f"Start time: {start_time}")
    logger.info(f"Input model: {model_file_path}")
    
    try:
        # Validate input model exists
        logger.info("🔍 Validating input model...")
        if not os.path.exists(model_file_path):
            error_msg = f"❌ CRITICAL ERROR: Input model file '{model_file_path}' not found!"
            logger.error(error_msg)
            logger.error("💡 HINT: Ensure the training step completed successfully")
            logger.error("💡 HINT: Check if model_file_path parameter is correct")
            logger.error("💡 HINT: Verify the training step outputs 'model.pt'")
            raise FileNotFoundError(error_msg)
        
        # Check model file size and validity
        model_size = os.path.getsize(model_file_path) / (1024 * 1024)  # MB
        logger.info(f"✅ Input model found: {model_file_path} ({model_size:.1f} MB)")
        
        if model_size < 1:  # Less than 1MB is suspicious
            logger.warning(f"⚠️  Model file seems small ({model_size:.1f} MB)")
            logger.warning("💡 HINT: This might indicate training issues")
        elif model_size > 500:  # Larger than 500MB is unusual
            logger.warning(f"⚠️  Model file seems large ({model_size:.1f} MB)")
            logger.warning("💡 HINT: This might indicate an issue with the model")

        # Validate YOLOv5 export module exists
        logger.info("🔍 Validating YOLOv5 export module...")
        try:
            from yolov5.export import run
            logger.info("✅ YOLOv5 export module available")
        except ImportError as e:
            error_msg = f"❌ CRITICAL ERROR: Cannot import YOLOv5 export module: {str(e)}"
            logger.error(error_msg)
            logger.error("💡 HINT: Ensure yolov5/* is included in pipeline dependencies")
            logger.error("💡 HINT: Check if YOLOv5 is properly installed")
            raise ImportError(error_msg)

        # Start conversion
        logger.info("🔄 Starting ONNX conversion...")
        logger.info("📋 Conversion parameters:")
        logger.info("  📊 Format: ONNX")
        logger.info("  🖼️  Image size: 640x640")
        logger.info("  🔧 ONNX opset: 13")
        
        conversion_start = datetime.now()
        
        try:
            run(
                weights=model_file_path,
                include=['onnx'],
                imgsz=(640, 640),
                opset=13,
                device='cpu',  # Force CPU for conversion to avoid GPU memory issues
                verbose=True,  # Enable verbose output
            )
        except Exception as e:
            error_msg = f"❌ CRITICAL ERROR: ONNX conversion failed: {str(e)}"
            logger.error(error_msg)
            logger.error("💡 HINT: Check if the input model is valid")
            logger.error("💡 HINT: Ensure sufficient disk space for conversion")
            logger.error("💡 HINT: Check if ONNX dependencies are installed")
            logger.error("💡 HINT: Try reducing image size if memory issues occur")
            raise RuntimeError(error_msg)
        
        conversion_end = datetime.now()
        conversion_duration = conversion_end - conversion_start
        logger.info(f"✅ Conversion completed in {conversion_duration}")

        # Validate output file
        logger.info("🔍 Validating conversion output...")
        expected_output = 'model.onnx'
        
        if not os.path.exists(expected_output):
            error_msg = f"❌ CRITICAL ERROR: Expected output file '{expected_output}' not found!"
            logger.error(error_msg)
            logger.error("💡 HINT: Check conversion logs above for errors")
            logger.error("💡 HINT: Ensure the conversion process completed successfully")
            
            # List available .onnx files for debugging
            onnx_files = []
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.onnx'):
                        onnx_files.append(os.path.join(root, file))
            
            if onnx_files:
                logger.error(f"💡 DEBUG: Found .onnx files: {onnx_files}")
                logger.error("💡 HINT: Check if output is in a different location")
            else:
                logger.error("💡 DEBUG: No .onnx files found in current directory")
            
            raise FileNotFoundError(error_msg)
        
        # Check output file size and validity
        output_size = os.path.getsize(expected_output) / (1024 * 1024)  # MB
        logger.info(f"✅ Output model created: {expected_output} ({output_size:.1f} MB)")
        
        # Validate output file is reasonable
        if output_size < 1:  # Less than 1MB is suspicious
            logger.warning(f"⚠️  Output model seems small ({output_size:.1f} MB)")
            logger.warning("💡 HINT: This might indicate conversion issues")
        
        # Compare sizes
        size_ratio = output_size / model_size
        logger.info(f"📊 Size comparison: Input {model_size:.1f} MB → Output {output_size:.1f} MB (ratio: {size_ratio:.2f})")
        
        if size_ratio > 2:  # ONNX significantly larger than PyTorch
            logger.warning("⚠️  ONNX model is significantly larger than PyTorch model")
            logger.warning("💡 HINT: This is sometimes normal but worth noting")
        elif size_ratio < 0.5:  # ONNX significantly smaller
            logger.warning("⚠️  ONNX model is significantly smaller than PyTorch model")
            logger.warning("💡 HINT: This might indicate conversion issues")

        # Log final statistics
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("MODEL CONVERSION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📁 Input: {model_file_path} ({model_size:.1f} MB)")
        logger.info(f"📁 Output: {expected_output} ({output_size:.1f} MB)")
        logger.info(f"⏱️  Total time: {total_duration}")
        logger.info(f"🔄 Conversion time: {conversion_duration}")
        logger.info(f"🏁 End time: {end_time}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ MODEL CONVERSION FAILED!")
        logger.error("=" * 60)
        logger.error(f"💥 Error: {str(e)}")
        logger.error(f"⏱️  Failed after: {datetime.now() - start_time}")
        logger.error("=" * 60)
        raise


if __name__ == '__main__':
    """
    Main execution block for standalone script usage.
    
    When run directly, this script converts the default 'model.pt' file
    to ONNX format. This is typically called as part of a Kubeflow pipeline
    after the model training step has completed.
    """
    convert_model()
