import os

from custom_logger import setup_logger

logger=setup_logger("util")

def create_folder_path(object_path):
    try:
        file_path=os.path.join(".",object_path)
        os.makedirs(file_path, exist_ok=True)
        logger.info(f"Create file derectory in {file_path}")

    except Exception as e:
        logger.error(f"file save path make error : {e}")