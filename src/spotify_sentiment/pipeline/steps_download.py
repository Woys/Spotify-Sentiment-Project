import os
from loguru import logger
from spotify_sentiment.pipeline.base import PipelineStep
from spotify_sentiment.core.config import settings

class DownloadStep(PipelineStep):
    @property
    def step_name(self) -> str: return "Data Ingestion"
    def execute(self):
        settings.DATA_DIR.mkdir(exist_ok=True, parents=True)
        if settings.RAW_DATA.exists(): return logger.info("Dataset found. Skipping.")
        os.environ['KAGGLE_USERNAME'] = settings.KAGGLE_USERNAME
        os.environ['KAGGLE_KEY'] = settings.KAGGLE_KEY
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(settings.KAGGLE_DATASET, path=settings.DATA_DIR, unzip=True)
        (settings.DATA_DIR / "top_podcasts.csv").rename(settings.RAW_DATA)
