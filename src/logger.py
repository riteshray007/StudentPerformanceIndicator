import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H')}.log"
LOG_FOLDER = f"{datetime.now().strftime('%m_%d_%Y')}"

logs_folder_path = os.path.join(os.getcwd(),"logs",LOG_FOLDER)
os.makedirs(logs_folder_path,exist_ok=True)

logs_path = os.path.join(logs_folder_path,LOG_FILE)

logging.basicConfig(
      filename=logs_path,
      format="[ %(asctime)s] %(lineno)d  %(name)s - %(levelname)s - %(message)s ",
      level = logging.INFO
)

# if __name__ == "__main__":
#       logging.info("logging has started ")
      
