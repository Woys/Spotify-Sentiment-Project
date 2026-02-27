from typing import List
from loguru import logger
from spotify_sentiment.pipeline.base import PipelineStep

class PipelineRunner:
    def __init__(self, steps: List[PipelineStep]): self.steps = steps
    def execute_all(self):
        for step in self.steps: step.run()
        logger.success("All pipeline steps executed successfully.")
