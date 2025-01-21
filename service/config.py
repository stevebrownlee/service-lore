""" Configuration settings for the Lore service """
import os
from pydantic import BaseModel, Field

class Settings(BaseModel):
    """ Configuration settings for the Lore service """
    # Valkey configuration
    VALKEY_HOST: str = Field(
        default=os.getenv("VALKEY_HOST", "localhost"),
        description="Valkey server hostname"
    )
    VALKEY_PORT: int = Field(
        default=int(os.getenv("VALKEY_PORT", "6379")),
        description="Valkey server port"
    )
    VALKEY_DB: int = Field(
        default=int(os.getenv("VALKEY_DB", "0")),
        description="Valkey database number"
    )

    # Model configuration
    MODEL_NAME: str = Field(
        default=os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1"),
        description="Name of the HuggingFace model to use"
    )
    MODEL_MAX_LENGTH: int = Field(
        default=int(os.getenv("MODEL_MAX_LENGTH", "500")),
        description="Maximum length of generated responses"
    )
    MODEL_TEMPERATURE: float = Field(
        default=float(os.getenv("MODEL_TEMPERATURE", "0.7")),
        description="Temperature for text generation"
    )

    # Metrics configuration
    PROMETHEUS_PORT: int = Field(
        default=int(os.getenv("PROMETHEUS_PORT", "8000")),
        description="Port for Prometheus metrics server"
    )

    # Channel names
    QUESTION_CHANNEL: str = Field(
        default="student_question",
        description="Valkey channel for incoming questions"
    )
    RESPONSE_CHANNEL_PREFIX: str = Field(
        default="lore_response_",
        description="Prefix for response channels"
    )
    ACK_CHANNEL: str = Field(
        default="lore_ack",
        description="Channel for acknowledgment messages"
    )

    # Response constraints
    MAX_RESPONSE_LENGTH: int = Field(
        default=int(os.getenv("MAX_RESPONSE_LENGTH", "1000")),
        description="Maximum length of complete response"
    )
    RESPONSE_TIMEOUT: int = Field(
        default=int(os.getenv("RESPONSE_TIMEOUT", "30")),
        description="Timeout for response generation in seconds"
    )
    BUFFER_MAX_AGE: float = Field(
        default=float(os.getenv("BUFFER_MAX_AGE", "300.0")),
        description="Maximum age of chunks in buffer before cleanup (seconds)"
    )
    BUFFER_WINDOW_SIZE: int = Field(
        default=int(os.getenv("BUFFER_WINDOW_SIZE", "5")),
        description="Maximum number of unacknowledged chunks in buffer"
    )

    # Log configuration
    LOG_LEVEL: str = Field(
        default=os.getenv("LOG_LEVEL", "INFO"),
        description="Logging level"
    )

    # Optional GPU configuration
    DEVICE: str = Field(
        default=os.getenv("DEVICE", "cpu"),
        description="Device to run model on (cpu or cuda)"
    )

    # System prompt
    SYSTEM_PROMPT: str = Field(
        default="""You are a programming concept explainer.
        You must ONLY provide natural language explanations.
        Never generate or include code examples.
        Never use code formatting, backticks, or code blocks.
        Explain concepts using analogies and plain language instead.
        The explanations must be generated to be understandable by a beginner.
        The explanations must assume the user has no prior knowledge of the concept.""",
        description="System prompt for the model"
    )

    class Config:
        """Pydantic model configuration"""
        env_file = ".env"