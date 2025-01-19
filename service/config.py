""" Configuration settings for the Lore service """
import os
from pydantic import BaseModel

class Settings(BaseModel):
    """ Configuration settings for the Lore service """
    # Valkey configuration
    VALKEY_HOST: str = os.getenv("VALKEY_HOST", "localhost")
    VALKEY_PORT: int = int(os.getenv("VALKEY_PORT", "6379"))
    VALKEY_DB: int = int(os.getenv("VALKEY_DB", "0"))

    # Model configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
    MODEL_MAX_LENGTH: int = int(os.getenv("MODEL_MAX_LENGTH", "500"))
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.7"))

    # Metrics configuration
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "8000"))

    # Channel names
    QUESTION_CHANNEL: str = "student_question"
    RESPONSE_CHANNEL_PREFIX: str = "lore_response_"

    # Response constraints
    MAX_RESPONSE_LENGTH: int = int(os.getenv("MAX_RESPONSE_LENGTH", "1000"))
    RESPONSE_TIMEOUT: int = int(os.getenv("RESPONSE_TIMEOUT", "30"))  # seconds

    # Log configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Optional GPU configuration
    DEVICE: str = os.getenv("DEVICE", "cpu")  # or "cuda" if GPU available

    class Config:
        env_file = ".env"