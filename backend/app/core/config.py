from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MWS GPT
    mws_gpt_base_url: str = "https://api.llmops.mts-corp.ru/v1"
    mws_gpt_api_key: str = ""

    # Models
    default_text_model: str = "gpt-4o-mini"
    fallback_text_model: str = "cotype-pro"
    vision_model: str = "gpt-4o"
    vision_fallback_model: str = "qwen2.5-vl-32b-instruct-awq"
    asr_model: str = "whisper-turbo-local-preview"
    image_generation_model: str = "sd3.5-large-image"
    image_generation_fallback_model: str = "sdxl-lightning-image"
    embedding_model: str = "bge-m3"

    # Infrastructure
    redis_url: str = "redis://redis:6379/0"
    qdrant_url: str = "http://qdrant:6333"
    postgres_dsn: str = "postgresql+asyncpg://postgres:change_me@postgres:5432/postgres"

    @property
    def postgres_dsn_sync(self) -> str:
        return self.postgres_dsn.replace("+asyncpg", "")

    # MinIO
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = ""
    minio_bucket: str = "ai-workspace"

    # Backend
    backend_host: str = "0.0.0.0"
    backend_reload: bool = False


settings = Settings()
