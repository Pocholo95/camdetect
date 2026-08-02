from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Captura
    RTSP_URL: str = "rtsp://user:pass@192.168.1.50:554/stream1"
    TARGET_FPS: float = 2
    MIN_FACE_PX: int = 50
    DET_SCORE_MIN: float = 0.60
    BLUR_VAR_MIN: float = 40

    # Matching
    SIM_THRESHOLD: float = 0.42
    MARGIN_THRESHOLD: float = 0.05
    UNKNOWN_CLUSTER_THRESHOLD: float = 0.35
    UNKNOWN_TTL_HOURS: float = 6
    UNKNOWN_RETENTION_DAYS: float = 30

    # Tracking / notificaciones
    MIN_FRAMES_CONFIRM: int = 4
    IOU_THRESHOLD: float = 0.3
    TRACK_MAX_AGE: int = 10
    PRESENCE_TIMEOUT_MIN: float = 5
    NOTIFY_COOLDOWN_MIN: float = 15
    BATCH_WINDOW_SEC: float = 60

    # Telegram
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # Infra
    DB_PATH: str = "./data/faces.db"
    MEDIA_DIR: str = "./data/media"
    WEB_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    ORT_NUM_THREADS: int = 2
    MEDIA_MAX_SIZE_MB: int = 2048


settings = Settings()
