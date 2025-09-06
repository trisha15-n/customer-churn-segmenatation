import logging
import os
from datetime import datetime
from src.logger import logging

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

log_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)
os.makedirs(os.path.dirname(log_path), exist_ok=True)

LOG_FILE_PATH  = os.path.join(os.getcwd(), 'logs', LOG_FILE)
logging.basicConfig(
    filename=log_path,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)