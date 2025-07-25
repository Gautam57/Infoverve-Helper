import logging
import os
from datetime import datetime


def setup_logger(caller_file: str):
    # Get the script name without extension
    script_name = os.path.splitext(os.path.basename(caller_file))[0]

    # Create logs/<script_name>/<date>.log
    log_file = f"{datetime.now().strftime('%m_%d_%Y')}.log"
    logs_dir = os.path.join(os.getcwd(), "logs", script_name)
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, log_file)

    # Setup logging
    logging.basicConfig(
        filename=log_file_path,
        format="[ %(asctime)s ] %(filename)s:%(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )