import gc
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from loguru import logger
from tqdm import tqdm
import time
import math
from spotify_sentiment.pipeline.base import PipelineStep
from spotify_sentiment.core.config import settings

class SentimentStep(PipelineStep):
    @property
    def step_name(self) -> str: return "Sentiment Analysis"
    
    def _normalize_score(self, text: str) -> float:
        if not isinstance(text, str) or not text.strip(): return 0.5
        return (self.analyzer.polarity_scores(text)['compound'] + 1.0) / 2.0

    def execute(self):
        try: nltk.data.find('sentiment/vader_lexicon.zip')
        except: nltk.download('vader_lexicon', quiet=True)
        self.analyzer = SentimentIntensityAnalyzer()
        
        cache = {}
        first_chunk = True
        
        with open(settings.RAW_DATA, 'rb') as f:
            total_rows = sum(1 for _ in f) - 1
        total_chunks = math.ceil(total_rows / settings.CHUNK_SIZE)
        
        with tqdm(total=total_chunks, desc="VADER Progress", unit="chk", dynamic_ncols=True) as pbar:
            for chunk in pd.read_csv(settings.RAW_DATA, chunksize=settings.CHUNK_SIZE, low_memory=False):
                t0 = time.time()
                chunk['description'] = chunk['description'].fillna("neutral")
                uniques = chunk['description'].unique()
                
                for t in [t for t in uniques if t not in cache]:
                    cache[t] = self._normalize_score(t)
                    
                chunk['sentiment_score'] = chunk['description'].map(cache)
                chunk.to_csv(settings.SENTIMENT_DATA, mode='a', index=False, header=first_chunk)
                
                first_chunk = False
                speed = settings.CHUNK_SIZE / max((time.time() - t0), 0.001)
                pbar.set_postfix({"Rows/sec": f"{speed:,.0f}", "Cache": len(cache)})
                pbar.update(1)
                
                if len(cache) > 200000: cache.clear()
                del chunk, uniques
                gc.collect()
