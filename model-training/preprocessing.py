"""
Data Preprocessing Pipeline Step for YOLOv5 Object Detection Training.

This module handles the preprocessing of custom training images for YOLOv5 object detection.
It organizes images and labels into train/validation/test splits required for model training.

The preprocessing step expects the following input structure:
    /data/
    ├── custom_training_images/
    │   └── <class_name>/
    │       ├── images/
    │       │   ├── image1.jpg
    │       │   └── image2.jpg
    │       └── labels/
    │           ├── image1.txt
    │           └── image2.txt
    └── configuration.yaml

And produces the following output structure:
    /data/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/

Author: Generated for Robot Hackathon Object Detection Pipeline
Version: 2.0 (Enhanced with comprehensive logging and error handling)
"""

from glob import glob
from math import floor
from os import makedirs, path
from shutil import copy
import yaml
import os
import logging
import sys
from datetime import datetime

from numpy import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def preprocess_data(data_folder='./data'):
    """
    Preprocess custom training images for YOLOv5 object detection training.
    
    This function takes raw training data organized by class and splits it into
    train/validation/test datasets with the proper directory structure expected
    by YOLOv5.
    
    Args:
        data_folder (str, optional): Path to the data directory containing
            custom_training_images and where output will be created.
            Defaults to './data'.
    
    Raises:
        FileNotFoundError: If required directories or files are missing.
        ValueError: If configuration file is invalid or missing required fields.
        RuntimeError: If file operations fail during dataset splitting.
    
    Directory Structure:
        Input:
            data_folder/
            ├── custom_training_images/
            │   └── <class_name>/
            │       ├── images/*.jpg
            │       └── labels/*.txt
            └── configuration.yaml
        
        Output:
            data_folder/
            ├── images/
            │   ├── train/
            │   ├── val/
            │   └── test/
            └── labels/
                ├── train/
                ├── val/
                └── test/
    
    Notes:
        - Uses a 75%/12.5%/12.5% split for train/validation/test
        - Random seed is set to 42 for reproducible splits
        - Class names are converted to lowercase for folder matching
        - Missing label files are logged as warnings but don't stop processing
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA PREPROCESSING")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    logger.info(f"Start time: {start_time}")
    logger.info(f"Data folder: {data_folder}")
    
    try:
        # Validate base data folder exists
        if not path.exists(data_folder):
            error_msg = f"❌ CRITICAL ERROR: Data folder '{data_folder}' does not exist!"
            logger.error(error_msg)
            logger.error("💡 HINT: Ensure the PVC is properly mounted to '/data' in your Kubeflow pipeline")
            logger.error("💡 HINT: Check if the 'object-detection-training-pvc' PVC exists and contains data")
            raise FileNotFoundError(error_msg)
        
        logger.info(f"✅ Base data folder '{data_folder}' exists")
        
        # Log current data folder contents
        try:
            data_contents = os.listdir(data_folder)
            logger.info(f"📁 Data folder contents: {data_contents}")
        except Exception as e:
            logger.warning(f"⚠️  Could not list data folder contents: {e}")

        # Create target folder structure
        logger.info("📂 Creating target folder structure...")
        folders_created = 0
        for folder in ['images', 'labels']:
            for split in ['train', 'val', 'test']:
                local_folder = f'{data_folder}/{folder}/{split}'
                if not path.exists(local_folder):
                    makedirs(local_folder)
                    folders_created += 1
                    logger.info(f"  ✅ Created: {local_folder}")
                else:
                    logger.info(f"  📁 Already exists: {local_folder}")
        
        logger.info(f"✅ Target folder structure ready ({folders_created} new folders created)")

        # Validate source data folder
        download_folder = f'{data_folder}/custom_training_images'
        if not path.exists(download_folder):
            error_msg = f"❌ CRITICAL ERROR: Source data folder '{download_folder}' does not exist!"
            logger.error(error_msg)
            logger.error("💡 HINT: Ensure your data ingestion step has populated the 'custom_training_images' folder")
            logger.error("💡 HINT: Expected structure: /data/custom_training_images/<class_name>/images/*.jpg")
            logger.error("💡 HINT: Expected structure: /data/custom_training_images/<class_name>/labels/*.txt")
            raise FileNotFoundError(error_msg)
        
        logger.info(f"✅ Source data folder '{download_folder}' exists")
        
        # Log source folder contents
        try:
            source_contents = os.listdir(download_folder)
            logger.info(f"📁 Source folder contents: {source_contents}")
        except Exception as e:
            logger.warning(f"⚠️  Could not list source folder contents: {e}")

        # Read and validate configuration
        logger.info("📋 Reading configuration...")
        class_labels = _read_class_labels('configuration.yaml')
        logger.info(f"✅ Configuration loaded - Classes: {class_labels}")

        # Validate class folders exist
        folder_names = [class_name.lower() for class_name in class_labels]
        logger.info(f"📂 Expected class folders: {folder_names}")
        
        for folder_name in folder_names:
            class_folder = f'{download_folder}/{folder_name}'
            if not path.exists(class_folder):
                error_msg = f"❌ CRITICAL ERROR: Class folder '{class_folder}' does not exist!"
                logger.error(error_msg)
                logger.error(f"💡 HINT: Create folder structure: {class_folder}/images/ and {class_folder}/labels/")
                logger.error(f"💡 HINT: Available folders: {os.listdir(download_folder) if path.exists(download_folder) else 'None'}")
                raise FileNotFoundError(error_msg)
            
            images_folder = f'{class_folder}/images'
            labels_folder = f'{class_folder}/labels'
            
            if not path.exists(images_folder):
                error_msg = f"❌ CRITICAL ERROR: Images folder '{images_folder}' does not exist!"
                logger.error(error_msg)
                logger.error(f"💡 HINT: Create the images folder: {images_folder}")
                raise FileNotFoundError(error_msg)
                
            if not path.exists(labels_folder):
                error_msg = f"❌ CRITICAL ERROR: Labels folder '{labels_folder}' does not exist!"
                logger.error(error_msg)
                logger.error(f"💡 HINT: Create the labels folder: {labels_folder}")
                raise FileNotFoundError(error_msg)
            
            logger.info(f"✅ Class folder structure valid: {folder_name}")

        # Collect image files for each class
        logger.info("🔍 Collecting image files...")
        images = []
        total_images = 0
        
        for folder_name in folder_names:
            image_files = _get_filenames(f'{download_folder}/{folder_name}/images')
            images.append(image_files)
            total_images += len(image_files)
            logger.info(f"  📸 {folder_name}: {len(image_files)} images")
        
        if total_images == 0:
            error_msg = "❌ CRITICAL ERROR: No images found in any class folder!"
            logger.error(error_msg)
            logger.error("💡 HINT: Add .jpg images to the /images/ subfolder of each class")
            logger.error("💡 HINT: Ensure images have corresponding .txt label files")
            raise FileNotFoundError(error_msg)
        
        logger.info(f"✅ Total images found: {total_images}")

        # Split datasets
        logger.info("🔀 Splitting datasets...")
        random.seed(42)
        train_ratio = 0.75
        val_ratio = 0.125
        test_ratio = 1 - train_ratio - val_ratio
        
        logger.info(f"📊 Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
        
        total_train = 0
        total_val = 0
        total_test = 0
        
        for i, image_set in enumerate(images):
            image_list = list(image_set)
            random.shuffle(image_list)
            train_size = floor(train_ratio * len(image_list))
            val_size = floor(val_ratio * len(image_list))
            test_size = len(image_list) - train_size - val_size
            
            logger.info(f"  📂 {folder_names[i]}: {train_size} train, {val_size} val, {test_size} test")
            
            _split_dataset(
                download_folder,
                data_folder,
                folder_names[i],
                image_list,
                train_size=train_size,
                val_size=val_size,
            )
            
            total_train += train_size
            total_val += val_size
            total_test += test_size

        # Log final statistics
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📊 Final dataset split:")
        logger.info(f"  🚂 Training: {total_train} images")
        logger.info(f"  ✅ Validation: {total_val} images")
        logger.info(f"  🧪 Test: {total_test} images")
        logger.info(f"  📈 Total: {total_train + total_val + total_test} images")
        logger.info(f"⏱️  Processing time: {duration}")
        logger.info(f"🏁 End time: {end_time}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ DATA PREPROCESSING FAILED!")
        logger.error("=" * 60)
        logger.error(f"💥 Error: {str(e)}")
        logger.error(f"⏱️  Failed after: {datetime.now() - start_time}")
        logger.error("=" * 60)
        raise


def _read_class_labels(configuration_file_path):
    """
    Read class labels from YAML configuration file with comprehensive validation.
    
    Args:
        configuration_file_path (str): Path to the YAML configuration file
            containing class definitions.
    
    Returns:
        list: List of class names as strings.
    
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration file is empty, invalid YAML,
            or missing required fields.
        KeyError: If the 'names' field is not found in the configuration.
        RuntimeError: If any other error occurs during file reading.
    
    Expected Configuration Format:
        ```yaml
        # Number of classes
        nc: 1
        
        # Class names
        names: ['Class1', 'Class2', ...]
        ```
    
    Notes:
        - The function validates YAML syntax and structure
        - Ensures at least one class name is defined
        - Provides detailed error messages for troubleshooting
    """
    try:
        if not path.exists(configuration_file_path):
            error_msg = f"❌ CRITICAL ERROR: Configuration file '{configuration_file_path}' not found!"
            logger.error(error_msg)
            logger.error("💡 HINT: Ensure 'configuration.yaml' is in the same directory as the script")
            logger.error("💡 HINT: Check the pipeline dependencies include 'configuration.yaml'")
            raise FileNotFoundError(error_msg)
        
        logger.info(f"📋 Reading configuration from: {configuration_file_path}")
        
        with open(configuration_file_path, 'r') as config_file:
            config = yaml.load(config_file.read(), Loader=yaml.SafeLoader)
        
        if config is None:
            error_msg = f"❌ CRITICAL ERROR: Configuration file '{configuration_file_path}' is empty or invalid YAML!"
            logger.error(error_msg)
            logger.error("💡 HINT: Check YAML syntax in configuration.yaml")
            raise ValueError(error_msg)
        
        if 'names' not in config:
            error_msg = "❌ CRITICAL ERROR: 'names' field not found in configuration!"
            logger.error(error_msg)
            logger.error("💡 HINT: Configuration should have 'names: [\"Class1\", \"Class2\"]' field")
            logger.error(f"💡 HINT: Available fields: {list(config.keys())}")
            raise KeyError(error_msg)
        
        class_labels = config['names']
        if not class_labels or len(class_labels) == 0:
            error_msg = "❌ CRITICAL ERROR: No class names defined in configuration!"
            logger.error(error_msg)
            logger.error("💡 HINT: Add class names to configuration.yaml: names: [\"YourClass\"]")
            raise ValueError(error_msg)
        
        logger.info(f"✅ Successfully loaded {len(class_labels)} class labels: {class_labels}")
        return class_labels
        
    except FileNotFoundError:
        raise
    except (yaml.YAMLError, KeyError, ValueError):
        raise
    except Exception as e:
        error_msg = f"❌ CRITICAL ERROR: Failed to read configuration: {str(e)}"
        logger.error(error_msg)
        logger.error("💡 HINT: Check file permissions and YAML syntax")
        raise RuntimeError(error_msg)


def _get_filenames(folder):
    """
    Scan a directory for JPG image files with comprehensive validation.
    
    Args:
        folder (str): Path to the directory to scan for image files.
    
    Returns:
        set: Set of image filenames (without path) found in the directory.
    
    Raises:
        FileNotFoundError: If the folder doesn't exist or no JPG images are found.
    
    Notes:
        - Only scans for .jpg files (case-sensitive)
        - Detects other image formats and provides helpful hints if found
        - Logs detailed debugging information for troubleshooting
        - Returns a set to avoid duplicate filenames
    
    Example:
        >>> filenames = _get_filenames('/path/to/images')
        >>> print(filenames)
        {'image1.jpg', 'image2.jpg', 'image3.jpg'}
    """
    logger.info(f"🔍 Scanning for images in: {folder}")
    
    if not path.exists(folder):
        error_msg = f"❌ CRITICAL ERROR: Image folder '{folder}' does not exist!"
        logger.error(error_msg)
        logger.error(f"💡 HINT: Create the folder: {folder}")
        raise FileNotFoundError(error_msg)
    
    filenames = set()
    jpg_pattern = path.join(folder, '*.jpg')
    jpg_files = glob(jpg_pattern)
    
    logger.info(f"📂 Searching pattern: {jpg_pattern}")
    
    for local_path in jpg_files:
        filename = path.split(local_path)[-1]
        filenames.add(filename)
        logger.debug(f"  📸 Found image: {filename}")
    
    if len(filenames) == 0:
        # Check for other image formats
        other_formats = []
        for ext in ['*.png', '*.jpeg', '*.JPG', '*.JPEG', '*.PNG']:
            other_files = glob(path.join(folder, ext))
            other_formats.extend([path.split(f)[-1] for f in other_files])
        
        error_msg = f"❌ CRITICAL ERROR: No .jpg images found in '{folder}'!"
        logger.error(error_msg)
        
        if other_formats:
            logger.error(f"💡 HINT: Found images in other formats: {other_formats[:5]}{'...' if len(other_formats) > 5 else ''}")
            logger.error("💡 HINT: Convert images to .jpg format or update the code to support other formats")
        else:
            logger.error("💡 HINT: Add .jpg images to this folder")
            logger.error("💡 HINT: Check if images are in the correct subdirectory")
        
        # List all files in the folder for debugging
        try:
            all_files = os.listdir(folder)
            logger.error(f"💡 DEBUG: All files in folder: {all_files[:10]}{'...' if len(all_files) > 10 else ''}")
        except Exception:
            pass
            
        raise FileNotFoundError(error_msg)
    
    logger.info(f"✅ Found {len(filenames)} .jpg images")
    return filenames


def _split_dataset(
        download_folder, data_folder, item, image_names, train_size, val_size):
    """
    Split a class dataset into train/validation/test sets with file copying.
    
    This function copies image and label files from the source class directory
    to the appropriate train/val/test subdirectories based on the specified
    split sizes.
    
    Args:
        download_folder (str): Path to the source data directory containing
            class subdirectories.
        data_folder (str): Path to the target directory where train/val/test
            subdirectories will be populated.
        item (str): Name of the class being processed (used as subdirectory name).
        image_names (list): List of image filenames to be split and copied.
        train_size (int): Number of images to allocate to training set.
        val_size (int): Number of images to allocate to validation set.
    
    Raises:
        FileNotFoundError: If source files or target directories don't exist.
        RuntimeError: If file copying operations fail.
    
    Notes:
        - Remaining images after train and val allocation go to test set
        - Copies both .jpg images and corresponding .txt label files
        - Missing label files are logged as warnings but don't stop processing
        - Provides detailed progress logging every 50 files
        - Validates all source and target paths before copying
    
    File Mapping:
        - image1.jpg -> image1.txt (label file)
        - Files are copied, not moved (originals remain intact)
    """
    
    logger.info(f"🔀 Splitting dataset for class: {item}")
    logger.info(f"  📊 Split sizes - Train: {train_size}, Val: {val_size}, Test: {len(image_names) - train_size - val_size}")
    
    copied_files = {'train': 0, 'val': 0, 'test': 0}
    missing_labels = []
    
    for i, image_name in enumerate(image_names):
        try:
            # Label filename
            label_name = image_name.replace('.jpg', '.txt')

            # Split into train, val, or test
            if i < train_size:
                split = 'train'
            elif i < train_size + val_size:
                split = 'val'
            else:
                split = 'test'

            # Source paths
            source_image_path = f'{download_folder}/{item}/images/{image_name}'
            source_label_path = f'{download_folder}/{item}/labels/{label_name}'

            # Validate source files exist
            if not path.exists(source_image_path):
                error_msg = f"❌ CRITICAL ERROR: Source image not found: {source_image_path}"
                logger.error(error_msg)
                logger.error(f"💡 HINT: Check if image '{image_name}' exists in {download_folder}/{item}/images/")
                raise FileNotFoundError(error_msg)
            
            if not path.exists(source_label_path):
                logger.warning(f"⚠️  Missing label file: {source_label_path}")
                missing_labels.append(label_name)
                # Continue processing but log the missing label

            # Destination paths
            target_image_folder = f'{data_folder}/images/{split}'
            target_label_folder = f'{data_folder}/labels/{split}'

            # Validate destination folders exist
            if not path.exists(target_image_folder):
                error_msg = f"❌ CRITICAL ERROR: Target image folder does not exist: {target_image_folder}"
                logger.error(error_msg)
                logger.error("💡 HINT: This should have been created earlier in the process")
                raise FileNotFoundError(error_msg)
            
            if not path.exists(target_label_folder):
                error_msg = f"❌ CRITICAL ERROR: Target label folder does not exist: {target_label_folder}"
                logger.error(error_msg)
                logger.error("💡 HINT: This should have been created earlier in the process")
                raise FileNotFoundError(error_msg)

            # Copy files
            try:
                copy(source_image_path, target_image_folder)
                if path.exists(source_label_path):
                    copy(source_label_path, target_label_folder)
                copied_files[split] += 1
                
                if (i + 1) % 50 == 0 or i == len(image_names) - 1:
                    logger.info(f"  📋 Progress: {i + 1}/{len(image_names)} files processed")
                    
            except Exception as e:
                error_msg = f"❌ CRITICAL ERROR: Failed to copy files for {image_name}: {str(e)}"
                logger.error(error_msg)
                logger.error(f"💡 HINT: Check disk space and file permissions")
                logger.error(f"💡 HINT: Source: {source_image_path}")
                logger.error(f"💡 HINT: Target: {target_image_folder}")
                raise RuntimeError(error_msg)
                
        except Exception as e:
            logger.error(f"❌ Failed to process {image_name}: {str(e)}")
            raise
    
    # Log summary
    logger.info(f"✅ Dataset split completed for {item}:")
    for split, count in copied_files.items():
        logger.info(f"  📂 {split}: {count} files")
    
    if missing_labels:
        logger.warning(f"⚠️  Missing {len(missing_labels)} label files for class {item}:")
        for label in missing_labels[:5]:  # Show first 5
            logger.warning(f"  📄 {label}")
        if len(missing_labels) > 5:
            logger.warning(f"  ... and {len(missing_labels) - 5} more")
        logger.warning("💡 HINT: Create corresponding .txt label files for these images")
        logger.warning("💡 HINT: Label files should contain: <class_id> <x_center> <y_center> <width> <height>")


if __name__ == '__main__':
    """
    Main execution block for standalone script usage.
    
    When run directly, this script processes data from the '/data' directory,
    which is the standard mount point in Kubeflow pipeline environments.
    """
    preprocess_data(data_folder='/data')
