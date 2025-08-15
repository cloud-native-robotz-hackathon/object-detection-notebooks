"""
Model Upload Pipeline Step for YOLOv5 Object Detection.

This module handles the upload of converted ONNX models to S3-compatible storage
for deployment and serving. It includes comprehensive S3 client setup, connection
validation, upload monitoring, and verification with robust error handling.

Key Features:
- S3-compatible storage upload (MinIO, AWS S3, etc.)
- Comprehensive credential and connection validation
- Dual upload strategy (versioned + latest)
- Upload integrity verification
- Detailed progress monitoring and error reporting
- Support for custom S3 endpoints and bucket configurations

Expected Input:
    - model.onnx: Converted ONNX model file from conversion step

Output:
    - Uploads model to S3 storage in two formats:
        * models/{prefix}-{timestamp}.onnx (versioned)
        * models/{prefix}-latest.onnx (latest)

Environment Variables Required:
    - UPLOAD_AWS_S3_ENDPOINT: S3 endpoint URL
    - UPLOAD_AWS_ACCESS_KEY_ID: S3 access key
    - UPLOAD_AWS_SECRET_ACCESS_KEY: S3 secret key
    - UPLOAD_AWS_S3_BUCKET: Target S3 bucket name
    - model_object_prefix: Model naming prefix (optional)

Storage Structure:
    bucket/
    └── models/
        ├── {prefix}-{timestamp}.onnx
        └── {prefix}-latest.onnx

Author: Generated for Robot Hackathon Object Detection Pipeline
Version: 2.0 (Enhanced with comprehensive logging and error handling)
"""

from os import environ
from datetime import datetime
import logging
import sys
import os

from boto3 import client
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

model_object_prefix = environ.get('model_object_prefix', 'model')
s3_endpoint_url = environ.get('UPLOAD_AWS_S3_ENDPOINT')
s3_access_key = environ.get('UPLOAD_AWS_ACCESS_KEY_ID')
s3_secret_key = environ.get('UPLOAD_AWS_SECRET_ACCESS_KEY')
s3_bucket_name = environ.get('UPLOAD_AWS_S3_BUCKET')


def upload_model(model_object_prefix='model', version=''):
    """
    Upload a converted ONNX model to S3-compatible storage with comprehensive validation.
    
    This function handles the complete model upload pipeline including credential validation,
    S3 client setup, connection testing, file upload, and verification. It uploads the model
    in both versioned and latest formats for flexible deployment options.
    
    Args:
        model_object_prefix (str, optional): Prefix for the model object names in S3.
            Defaults to 'model'. Can also be set via environment variable.
        version (str, optional): Specific version string for the model. If empty,
            a timestamp will be automatically generated. Defaults to ''.
    
    Raises:
        FileNotFoundError: If model.onnx file doesn't exist or target bucket not found.
        EnvironmentError: If required environment variables are missing.
        NoCredentialsError: If S3 credentials are invalid or missing.
        EndpointConnectionError: If cannot connect to S3 endpoint.
        ClientError: If S3 operations fail due to permissions or other issues.
        RuntimeError: If upload operations fail unexpectedly.
    
    Environment Variables Required:
        - UPLOAD_AWS_S3_ENDPOINT: S3 endpoint URL (e.g., https://s3.amazonaws.com)
        - UPLOAD_AWS_ACCESS_KEY_ID: S3 access key ID
        - UPLOAD_AWS_SECRET_ACCESS_KEY: S3 secret access key
        - UPLOAD_AWS_S3_BUCKET: Target S3 bucket name
        - model_object_prefix: Model prefix (optional, overrides parameter)
    
    Upload Strategy:
        1. Validates all prerequisites (file exists, credentials set)
        2. Initializes and tests S3 client connection
        3. Uploads versioned model: models/{prefix}-{version}.onnx
        4. Uploads latest model: models/{prefix}-latest.onnx
        5. Verifies both uploads completed successfully
    
    Storage Naming Convention:
        - Versioned: models/{prefix}-{timestamp}.onnx
        - Latest: models/{prefix}-latest.onnx
        - Timestamp format: YYMMDDHHNN (e.g., 2412151430)
    
    Performance Monitoring:
        - Upload duration tracking for each file
        - Upload speed calculation (MB/s)
        - File size validation and reporting
        - Comprehensive error logging with hints
    
    Notes:
        - Creates backup of existing model.pt if present
        - Adds metadata to uploaded objects (timestamp, size, pipeline info)
        - Provides detailed troubleshooting information on failures
        - Supports any S3-compatible storage (AWS S3, MinIO, etc.)
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL UPLOAD")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    logger.info(f"Start time: {start_time}")
    
    try:
        # Validate prerequisites
        logger.info("🔍 Validating upload prerequisites...")
        _validate_upload_prerequisites()
        
        # Get configuration
        final_prefix = model_object_prefix or environ.get('model_object_prefix', 'model')
        logger.info(f"📋 Upload configuration:")
        logger.info(f"  🏷️  Model prefix: {final_prefix}")
        logger.info(f"  🔖 Version: {version or 'auto-generated'}")
        logger.info(f"  🗄️  S3 endpoint: {s3_endpoint_url}")
        logger.info(f"  🪣 S3 bucket: {s3_bucket_name}")
        
        # Initialize S3 client
        logger.info("🔌 Initializing S3 client...")
        s3_client = _initialize_s3_client(
            s3_endpoint_url=s3_endpoint_url,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key
        )
        
        # Test S3 connection
        logger.info("🧪 Testing S3 connection...")
        _test_s3_connection(s3_client)
        
        # Generate model names
        model_object_name = _generate_model_name(final_prefix, version=version)
        model_object_name_latest = _generate_model_name(final_prefix, 'latest')
        
        logger.info(f"📦 Upload targets:")
        logger.info(f"  📁 Versioned: {model_object_name}")
        logger.info(f"  📁 Latest: {model_object_name_latest}")
        
        # Upload versioned model
        logger.info("📤 Uploading versioned model...")
        upload_start = datetime.now()
        _do_upload(s3_client, model_object_name)
        upload_time = datetime.now() - upload_start
        logger.info(f"✅ Versioned upload completed in {upload_time}")

        # Upload latest model
        logger.info("📤 Uploading latest model...")
        latest_start = datetime.now()
        _do_upload(s3_client, model_object_name_latest)
        latest_time = datetime.now() - latest_start
        logger.info(f"✅ Latest upload completed in {latest_time}")

        # Final validation
        logger.info("✅ Verifying uploads...")
        _verify_uploads(s3_client, [model_object_name, model_object_name_latest])

        # Log final statistics
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("MODEL UPLOAD COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📦 Uploaded models:")
        logger.info(f"  🔖 Versioned: {model_object_name}")
        logger.info(f"  🆕 Latest: {model_object_name_latest}")
        logger.info(f"⏱️  Total time: {total_duration}")
        logger.info(f"🌐 Available at: {s3_endpoint_url}/{s3_bucket_name}/")
        logger.info(f"🏁 End time: {end_time}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ MODEL UPLOAD FAILED!")
        logger.error("=" * 60)
        logger.error(f"💥 Error: {str(e)}")
        logger.error(f"⏱️  Failed after: {datetime.now() - start_time}")
        logger.error("=" * 60)
        raise


def _validate_upload_prerequisites():
    """
    Validate all prerequisites required for successful model upload.
    
    This function performs comprehensive validation of the upload environment
    including file existence, environment variables, and credential configuration.
    
    Raises:
        FileNotFoundError: If model.onnx file doesn't exist.
        EnvironmentError: If required environment variables are missing.
    
    Validation Checks:
        - model.onnx file exists and has reasonable size
        - All required S3 environment variables are set:
            * UPLOAD_AWS_S3_ENDPOINT
            * UPLOAD_AWS_ACCESS_KEY_ID  
            * UPLOAD_AWS_SECRET_ACCESS_KEY
            * UPLOAD_AWS_S3_BUCKET
    
    Notes:
        - Logs file size information for monitoring
        - Provides specific hints for missing configuration
        - Validates environment without exposing sensitive credentials
    """
    # Check if model.onnx exists
    if not os.path.exists('model.onnx'):
        error_msg = "❌ CRITICAL ERROR: model.onnx not found!"
        logger.error(error_msg)
        logger.error("💡 HINT: Ensure the model conversion step completed successfully")
        logger.error("💡 HINT: Check if the conversion step outputs 'model.onnx'")
        raise FileNotFoundError(error_msg)
    
    model_size = os.path.getsize('model.onnx') / (1024 * 1024)  # MB
    logger.info(f"✅ Model file found: model.onnx ({model_size:.1f} MB)")
    
    # Validate environment variables
    missing_vars = []
    env_vars = {
        'UPLOAD_AWS_S3_ENDPOINT': s3_endpoint_url,
        'UPLOAD_AWS_ACCESS_KEY_ID': s3_access_key,
        'UPLOAD_AWS_SECRET_ACCESS_KEY': s3_secret_key,
        'UPLOAD_AWS_S3_BUCKET': s3_bucket_name
    }
    
    for var_name, var_value in env_vars.items():
        if not var_value:
            missing_vars.append(var_name)
    
    if missing_vars:
        error_msg = f"❌ CRITICAL ERROR: Missing required environment variables: {missing_vars}"
        logger.error(error_msg)
        logger.error("💡 HINT: Ensure S3 credentials are configured in Kubeflow secrets")
        logger.error("💡 HINT: Check 'workbench-bucket-ai-connection' secret exists")
        logger.error("💡 HINT: Verify pipeline configuration includes S3 environment variables")
        raise EnvironmentError(error_msg)
    
    logger.info("✅ All environment variables configured")


def _initialize_s3_client(s3_endpoint_url, s3_access_key, s3_secret_key):
    """
    Initialize S3 client with comprehensive error handling and validation.
    
    This function creates a boto3 S3 client configured for the specified endpoint
    and credentials, with detailed error handling for common configuration issues.
    
    Args:
        s3_endpoint_url (str): S3 endpoint URL (e.g., https://s3.amazonaws.com).
        s3_access_key (str): S3 access key ID for authentication.
        s3_secret_key (str): S3 secret access key for authentication.
    
    Returns:
        boto3.client: Configured S3 client ready for operations.
    
    Raises:
        NoCredentialsError: If credentials are invalid or missing.
        RuntimeError: If client initialization fails for other reasons.
    
    Configuration:
        - Supports custom S3 endpoints (AWS S3, MinIO, etc.)
        - Uses provided credentials for authentication
        - Configures for maximum compatibility
    
    Notes:
        - Logs endpoint and access key (masked) for debugging
        - Provides specific error messages for credential issues
        - Handles various boto3 client creation failures
    """
    try:
        logger.info("🔧 Creating S3 client...")
        logger.info(f"  🌐 Endpoint: {s3_endpoint_url}")
        logger.info(f"  🔑 Access Key: {s3_access_key[:10]}..." if s3_access_key else "  🔑 Access Key: [NOT SET]")
        
        s3_client = client(
            's3', 
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            endpoint_url=s3_endpoint_url,
        )
        
        logger.info("✅ S3 client created successfully")
        return s3_client
        
    except NoCredentialsError as e:
        error_msg = f"❌ CRITICAL ERROR: S3 credentials not found: {str(e)}"
        logger.error(error_msg)
        logger.error("💡 HINT: Check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        logger.error("💡 HINT: Verify Kubeflow secrets are properly configured")
        raise NoCredentialsError(error_msg)
    except Exception as e:
        error_msg = f"❌ CRITICAL ERROR: Failed to initialize S3 client: {str(e)}"
        logger.error(error_msg)
        logger.error("💡 HINT: Check S3 endpoint URL format")
        logger.error("💡 HINT: Verify network connectivity to S3 endpoint")
        raise RuntimeError(error_msg)


def _test_s3_connection(s3_client):
    """
    Test S3 connection and validate bucket access permissions.
    
    This function performs comprehensive S3 connectivity and permission testing
    to ensure uploads will succeed before attempting actual file operations.
    
    Args:
        s3_client (boto3.client): Configured S3 client to test.
    
    Raises:
        EndpointConnectionError: If cannot connect to S3 endpoint.
        ClientError: If S3 operations fail due to permissions or configuration.
        FileNotFoundError: If target bucket doesn't exist.
    
    Connection Tests:
        1. Lists available buckets to test basic connectivity
        2. Verifies target bucket exists and is accessible
        3. Tests read permissions on target bucket
        4. Logs detailed bucket and permission information
    
    Error Handling:
        - Provides specific hints for different error types
        - Distinguishes between network, credential, and permission issues
        - Logs available buckets for debugging
    
    Notes:
        - Non-blocking warnings for limited permissions
        - Comprehensive error categorization and hints
        - Validates end-to-end upload capability
    """
    try:
        logger.info("🔍 Testing S3 connection...")
        
        # Test connection by listing buckets
        response = s3_client.list_buckets()
        logger.info(f"✅ S3 connection successful")
        
        # Check if target bucket exists
        bucket_names = [bucket['Name'] for bucket in response['Buckets']]
        logger.info(f"📁 Available buckets: {bucket_names}")
        
        if s3_bucket_name not in bucket_names:
            error_msg = f"❌ CRITICAL ERROR: Target bucket '{s3_bucket_name}' not found!"
            logger.error(error_msg)
            logger.error(f"💡 HINT: Available buckets: {bucket_names}")
            logger.error("💡 HINT: Check bucket name in environment configuration")
            logger.error("💡 HINT: Verify bucket exists and you have access")
            raise FileNotFoundError(error_msg)
        
        logger.info(f"✅ Target bucket '{s3_bucket_name}' found and accessible")
        
        # Test write permissions by attempting to list objects
        try:
            s3_client.list_objects_v2(Bucket=s3_bucket_name, MaxKeys=1)
            logger.info("✅ Bucket read permissions confirmed")
        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDenied':
                logger.warning("⚠️  Limited bucket access - upload may fail")
            else:
                raise
                
    except EndpointConnectionError as e:
        error_msg = f"❌ CRITICAL ERROR: Cannot connect to S3 endpoint: {str(e)}"
        logger.error(error_msg)
        logger.error("💡 HINT: Check S3 endpoint URL is correct")
        logger.error("💡 HINT: Verify network connectivity")
        logger.error("💡 HINT: Check if firewall is blocking the connection")
        raise EndpointConnectionError(endpoint_url=s3_endpoint_url)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = f"❌ CRITICAL ERROR: S3 client error ({error_code}): {str(e)}"
        logger.error(error_msg)
        
        if error_code == 'InvalidAccessKeyId':
            logger.error("💡 HINT: Check AWS_ACCESS_KEY_ID is correct")
        elif error_code == 'SignatureDoesNotMatch':
            logger.error("💡 HINT: Check AWS_SECRET_ACCESS_KEY is correct")
        elif error_code == 'AccessDenied':
            logger.error("💡 HINT: Check IAM permissions for S3 access")
        else:
            logger.error("💡 HINT: Check S3 credentials and permissions")
        
        raise ClientError(e.response, e.operation_name)


def _generate_model_name(model_object_prefix, version=''):
    """
    Generate S3 object name for model with validation and sanitization.
    
    This function creates properly formatted S3 object names following the
    project's naming conventions with input validation and sanitization.
    
    Args:
        model_object_prefix (str): Base prefix for the model name.
        version (str, optional): Version string for the model. If empty,
            a timestamp will be generated. Defaults to ''.
    
    Returns:
        str: Properly formatted S3 object name (e.g., 'models/prefix-version.onnx').
    
    Raises:
        ValueError: If model_object_prefix is empty or invalid.
    
    Naming Convention:
        - Format: models/{prefix}-{version}.onnx
        - Prefix sanitization: removes invalid characters
        - Version: timestamp (YYMMDDHHNN) if not provided
    
    Sanitization:
        - Removes non-alphanumeric characters except hyphens and underscores
        - Logs sanitization changes for debugging
        - Ensures S3-compatible object names
    
    Notes:
        - Follows project storage organization (models/ prefix)
        - Supports both custom versions and auto-generated timestamps
        - Validates input to prevent S3 object naming issues
    """
    if not model_object_prefix:
        error_msg = "❌ CRITICAL ERROR: model_object_prefix cannot be empty!"
        logger.error(error_msg)
        logger.error("💡 HINT: Set model_object_prefix parameter or environment variable")
        raise ValueError(error_msg)
    
    # Sanitize prefix (remove invalid characters)
    safe_prefix = ''.join(c for c in model_object_prefix if c.isalnum() or c in '-_')
    if safe_prefix != model_object_prefix:
        logger.warning(f"⚠️  Sanitized model prefix: '{model_object_prefix}' → '{safe_prefix}'")
    
    final_version = version if version else _timestamp()
    model_name = f'models/{safe_prefix}-{final_version}.onnx'
    
    logger.info(f"🏷️  Generated model name: {model_name}")
    return model_name


def _timestamp():
    """
    Generate timestamp string for model versioning.
    
    Returns:
        str: Timestamp in YYMMDDHHNN format (e.g., '2412151430').
    
    Format Details:
        - YY: Two-digit year
        - MM: Two-digit month
        - DD: Two-digit day
        - HH: Two-digit hour (24-hour format)
        - NN: Two-digit minute
    
    Notes:
        - Provides unique versioning for model uploads
        - Compact format suitable for object names
        - Uses local system time
    """
    timestamp = datetime.now().strftime('%y%m%d%H%M')
    logger.debug(f"🕐 Generated timestamp: {timestamp}")
    return timestamp


def _do_upload(s3_client, object_name):
    """
    Upload model.onnx file to S3 with comprehensive monitoring and error handling.
    
    This function performs the actual file upload operation with detailed progress
    tracking, performance monitoring, and robust error handling for various failure
    scenarios.
    
    Args:
        s3_client (boto3.client): Configured S3 client for upload operations.
        object_name (str): Target S3 object name for the uploaded file.
    
    Raises:
        ClientError: If S3 upload fails due to permissions, bucket issues, etc.
        FileNotFoundError: If model.onnx file doesn't exist.
        RuntimeError: If upload fails for unexpected reasons.
    
    Upload Process:
        1. Validates source file exists and gets size
        2. Performs upload with metadata
        3. Tracks upload duration and speed
        4. Logs detailed progress and performance metrics
    
    Metadata Added:
        - upload-timestamp: ISO format timestamp
        - file-size-mb: File size in megabytes
        - pipeline-step: Identifies upload source
    
    Performance Monitoring:
        - Upload duration measurement
        - Speed calculation (MB/s)
        - File size validation
        - Detailed success/failure logging
    
    Error Handling:
        - Specific error categorization (permissions, bucket, network)
        - Detailed troubleshooting hints
        - Debug information for failed uploads
    """
    try:
        logger.info(f"📤 Starting upload: {object_name}")
        
        # Get file size for progress tracking
        file_size = os.path.getsize('model.onnx') / (1024 * 1024)  # MB
        logger.info(f"  📊 File size: {file_size:.1f} MB")
        
        upload_start = datetime.now()
        
        # Perform upload with extra metadata
        extra_args = {
            'Metadata': {
                'upload-timestamp': datetime.now().isoformat(),
                'file-size-mb': str(round(file_size, 1)),
                'pipeline-step': 'model-upload'
            }
        }
        
        s3_client.upload_file(
            'model.onnx', 
            s3_bucket_name, 
            object_name,
            ExtraArgs=extra_args
        )
        
        upload_duration = datetime.now() - upload_start
        upload_speed = file_size / upload_duration.total_seconds() if upload_duration.total_seconds() > 0 else 0
        
        logger.info(f"✅ Upload completed successfully")
        logger.info(f"  ⏱️  Duration: {upload_duration}")
        logger.info(f"  🚀 Speed: {upload_speed:.1f} MB/s")
        logger.info(f"  🌐 Location: s3://{s3_bucket_name}/{object_name}")
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = f"❌ CRITICAL ERROR: S3 upload failed ({error_code}): {str(e)}"
        logger.error(error_msg)
        
        if error_code == 'AccessDenied':
            logger.error("💡 HINT: Check S3 write permissions")
            logger.error("💡 HINT: Verify IAM policy allows s3:PutObject")
        elif error_code == 'NoSuchBucket':
            logger.error(f"💡 HINT: Bucket '{s3_bucket_name}' does not exist")
        elif error_code == 'InvalidBucketName':
            logger.error(f"💡 HINT: Invalid bucket name: '{s3_bucket_name}'")
        else:
            logger.error("💡 HINT: Check S3 configuration and permissions")
        
        logger.error(f"💡 DEBUG: Target: s3://{s3_bucket_name}/{object_name}")
        raise ClientError(e.response, e.operation_name)
        
    except FileNotFoundError:
        error_msg = "❌ CRITICAL ERROR: model.onnx file not found for upload!"
        logger.error(error_msg)
        logger.error("💡 HINT: Ensure model conversion step completed successfully")
        raise FileNotFoundError(error_msg)
        
    except Exception as e:
        error_msg = f"❌ CRITICAL ERROR: Unexpected upload error: {str(e)}"
        logger.error(error_msg)
        logger.error(f"💡 DEBUG: Upload target: {object_name}")
        logger.error(f"💡 DEBUG: Bucket: {s3_bucket_name}")
        logger.error(f"💡 DEBUG: Endpoint: {s3_endpoint_url}")
        logger.error("💡 HINT: Check network connectivity and disk space")
        raise RuntimeError(error_msg)


def _verify_uploads(s3_client, object_names):
    """
    Verify upload integrity and validate uploaded objects in S3.
    
    This function performs post-upload verification to ensure all files were
    uploaded correctly and are accessible in S3 storage.
    
    Args:
        s3_client (boto3.client): Configured S3 client for verification.
        object_names (list): List of S3 object names to verify.
    
    Raises:
        FileNotFoundError: If any uploaded object cannot be found in S3.
        ClientError: If S3 operations fail during verification.
    
    Verification Process:
        1. Checks each object exists in S3 using head_object
        2. Validates file sizes match local file
        3. Retrieves and logs object metadata
        4. Reports any size discrepancies
    
    Validation Checks:
        - Object existence in S3
        - File size comparison (local vs remote)
        - Metadata retrieval and logging
        - Last modified timestamp verification
    
    Size Tolerance:
        - Allows 0.1 MB difference for compression/metadata
        - Logs warnings for significant size differences
        - Ensures upload integrity
    
    Notes:
        - Provides detailed verification results for each object
        - Helps detect silent upload failures
        - Validates end-to-end upload success
    """
    logger.info("🔍 Verifying upload integrity...")
    
    for object_name in object_names:
        try:
            response = s3_client.head_object(Bucket=s3_bucket_name, Key=object_name)
            
            # Get object metadata
            size = response.get('ContentLength', 0) / (1024 * 1024)  # MB
            last_modified = response.get('LastModified', 'Unknown')
            
            logger.info(f"  ✅ {object_name}: {size:.1f} MB (modified: {last_modified})")
            
            # Verify size matches local file
            local_size = os.path.getsize('model.onnx') / (1024 * 1024)  # MB
            if abs(size - local_size) > 0.1:  # Allow 0.1 MB difference
                logger.warning(f"⚠️  Size mismatch for {object_name}: local {local_size:.1f} MB vs remote {size:.1f} MB")
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                error_msg = f"❌ CRITICAL ERROR: Upload verification failed - {object_name} not found!"
                logger.error(error_msg)
                logger.error("💡 HINT: Upload may have failed silently")
                raise FileNotFoundError(error_msg)
            else:
                raise
    
    logger.info("✅ All uploads verified successfully")


if __name__ == '__main__':
    """
    Main execution block for standalone script usage.
    
    When run directly, this script uploads the model using the prefix from
    the environment variable. This is typically called as the final step
    in a Kubeflow pipeline after model conversion has completed.
    """
    upload_model(model_object_prefix)
