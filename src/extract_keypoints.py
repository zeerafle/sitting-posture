import os
import sys
import time
import tensorflow_hub as hub
from loguru import logger

from keypoints_extraction.preprocessor import MoveNetPreprocessor

# Configure Loguru
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_path, exist_ok=True)
logger.remove()  # Remove default handler
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add(os.path.join(log_path, "keypoints_extraction_{time}.log"), rotation="500 MB", retention="30 days", compression="zip", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")

# Set up paths
ORIGINAL_DIR = "/teamspace/studios/01-data-download-kaggle/.cache/kagglehub/datasets/zeerafle/sitting-posture/versions/6"
DATA_DIR = "data"
IMAGES_OUT_DIR = os.path.join(DATA_DIR, "poses_images_out")
CSVS_OUT = os.path.join(DATA_DIR, "data.csv")

logger.info("Starting keypoint extraction process")
logger.debug(f"Source directory: {ORIGINAL_DIR}")
logger.debug(f"Output images directory: {IMAGES_OUT_DIR}")
logger.debug(f"Output CSV path: {CSVS_OUT}")

# Ensure output directories exist
os.makedirs(IMAGES_OUT_DIR, exist_ok=True)
logger.debug(f"Ensured output directory exists: {IMAGES_OUT_DIR}")

# Load model
logger.info("Loading MoveNet model...")
start_time = time.time()
try:
    model = hub.load(
        "https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-thunder/4"
    )
    movenet = model.signatures["serving_default"]
    load_time = time.time() - start_time
    logger.success(f"MoveNet model loaded successfully in {load_time:.2f} seconds")
except Exception as e:
    logger.error(f"Failed to load MoveNet model: {str(e)}")
    sys.exit(1)

# Initialize preprocessor
logger.info("Initializing MoveNet preprocessor")
try:
    preprocessor = MoveNetPreprocessor(
        model=movenet,
        images_in_folder=os.path.join(ORIGINAL_DIR),
        images_out_folder=IMAGES_OUT_DIR,
        csvs_out_path=CSVS_OUT,
        batch_size=4,
    )
    logger.debug("Preprocessor initialized with batch size 4")
except Exception as e:
    logger.error(f"Failed to initialize preprocessor: {str(e)}")
    sys.exit(1)

# Process images
logger.info("Processing images and extracting landmarks...")
start_time = time.time()
try:
    preprocessor.process()
    process_time = time.time() - start_time
    logger.success(f"Image processing completed in {process_time:.2f} seconds")

    # Check if output CSV exists
    if os.path.exists(CSVS_OUT):
        file_size = os.path.getsize(CSVS_OUT) / (1024 * 1024)  # Convert to MB
        logger.info(f"Generated CSV file: {CSVS_OUT} ({file_size:.2f} MB)")
    else:
        logger.warning(f"Expected output file {CSVS_OUT} not found")

except Exception as e:
    logger.error(f"Error during image processing: {str(e)}", exc_info=True)
    sys.exit(1)

logger.info("Keypoint extraction process completed")
