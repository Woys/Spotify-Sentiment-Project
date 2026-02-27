import argparse, sys
from loguru import logger
from spotify_sentiment.pipeline.runner import PipelineRunner
from spotify_sentiment.pipeline.steps_download import DownloadStep
from spotify_sentiment.pipeline.steps_sentiment import SentimentStep
from spotify_sentiment.pipeline.steps_analyze import AnalyzeStep
from spotify_sentiment.pipeline.steps_visualize import VisualizeStep

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["download", "sentiment", "analyze", "visualize", "all"], default="all")
    args = parser.parse_args()
    s = {"download": DownloadStep(), "sentiment": SentimentStep(), "analyze": AnalyzeStep(), "visualize": VisualizeStep()}
    try: PipelineRunner(list(s.values()) if args.step == "all" else [s[args.step]]).execute_all()
    except Exception as e: logger.critical(e); sys.exit(1)
if __name__ == "__main__": main()
