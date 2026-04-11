import sys, os
import uvicorn
from openenv.core.env_server import create_fastapi_app

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_cleaning_env import DataCleaningEnv, DataCleanerAction, DataCleanerObservation

app = create_fastapi_app(DataCleaningEnv, DataCleanerAction, DataCleanerObservation)

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
