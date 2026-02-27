from abc import ABC, abstractmethod
import psutil
import time
from loguru import logger

class PipelineStep(ABC):
    @property
    @abstractmethod
    def step_name(self) -> str: pass

    @abstractmethod
    def execute(self) -> None: pass

    def log_telemetry(self, ctx: str = ""):
        mem = psutil.virtual_memory()
        logger.debug(f"[{self.step_name} RAM] {ctx} | Used: {mem.percent}% ({mem.used / 1024**3:.2f} GB)")

    def run(self) -> None:
        logger.info(f"Starting Phase: {self.step_name}")
        start_time = time.time()
        self.log_telemetry("Start")
        try:
            self.execute()
            duration = time.time() - start_time
            logger.success(f"Completed Phase: {self.step_name} in {duration:.2f}s")
            self.log_telemetry("End")
        except Exception as e:
            logger.error(f"Failed Phase {self.step_name}: {e}")
            raise
