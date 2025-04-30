import uvicorn
from config.config import APP_HOST, APP_PORT, APP_RELOAD
import logging

from config.logging_config import CustomFormatter

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())

logger.addHandler(handler)

if __name__ == "__main__":
    logging.info("Starting server...")
    uvicorn.run(
        "src.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_RELOAD)
    logging.info("Server started")