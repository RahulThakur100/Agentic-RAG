import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_CONN_STRING = os.getenv("DB_CONN_STRING")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
