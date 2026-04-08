import sys
import os
import uvicorn
from openenv.core.env_server import create_fastapi_app

# Add root to sys.path so it can find data_cleaning_env
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_cleaning_env import DataCleaningEnv

app = create_fastapi_app(DataCleaningEnv)

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
