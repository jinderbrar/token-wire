from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    LLM_MODEL_NAME: str = "qwen2.5-coder:1.5b"

    # Allowed models - only these will be shown in the frontend
    # Add models here as you generate dictionaries for them
    ALLOWED_MODELS: list[str] = [
        "qwen2.5-coder:1.5b",
        "qwen2.5-coder:7b",
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
