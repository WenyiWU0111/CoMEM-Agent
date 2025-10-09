"""Logging setup for the GUI Agent"""
import logging
from pathlib import Path
import argparse


def setup_logging(args: argparse.Namespace):
    """Setup logging configuration and return logging components"""
    LOG_FOLDER = f"log_files/{args.evaluation_type}/{args.domain}"
    Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
    datetime = args.datetime
    LOG_FILE_NAME = f"{LOG_FOLDER}/log_{args.model}_{datetime}.log"
    
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_FILE_NAME)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # Set the log format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    return datetime, LOG_FILE_NAME, logger 