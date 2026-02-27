from pathlib import Path
from typing import Dict, List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    BASE_DIR: Path = Path.cwd()
    DATA_DIR: Path = BASE_DIR / "data"
    ASSETS_DIR: Path = BASE_DIR / "assets"
    RAW_DATA: Path = DATA_DIR / "spotify_podcasts.csv"
    SENTIMENT_DATA: Path = DATA_DIR / "sentiment_results.csv"
    TOPIC_METRICS: Path = DATA_DIR / "topic_metrics.csv"
    WORD_METRICS: Path = DATA_DIR / "word_metrics.csv"

    KAGGLE_USERNAME: str = ""
    KAGGLE_KEY: str = ""
    KAGGLE_DATASET: str = "daniilmiheev/top-spotify-podcasts-daily-updated"

    USE_EXACT_MATCH_ONLY: bool = False
    CHUNK_SIZE: int = 50000

    TOPIC_DEFINITIONS: Dict[str, List[str]] = {
        "Artificial Intelligence": ["ai", "machine learning", "deep learning", "algorithm", "neural network"],
        "Climate Change": ["climate", "global warming", "sustainability", "emissions", "carbon"],
        "Economy": ["economy", "inflation", "recession", "markets", "interest rates"],
        "Startup": ["startup", "founder", "venture capital", "entrepreneur", "funding"],
        "Nutrition": ["nutrition", "diet", "protein", "vitamins", "healthy eating"]
    }

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
