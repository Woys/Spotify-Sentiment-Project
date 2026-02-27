import gc
import time
import math
import psutil
import pandas as pd
import gensim.downloader as api
from loguru import logger
from tqdm import tqdm
from spotify_sentiment.pipeline.base import PipelineStep
from spotify_sentiment.core.config import settings
import fast_scanner

class AnalyzeStep(PipelineStep):
    @property
    def step_name(self) -> str: return "C++ Hash Extraction"

    def __init__(self):
        self.glove = None if settings.USE_EXACT_MATCH_ONLY else api.load('glove-wiki-gigaword-50')

    def execute(self) -> None:
        patterns = {}
        for topic, seeds in settings.TOPIC_DEFINITIONS.items():
            vocab = set(w.lower() for w in seeds)
            if self.glove:
                for word in seeds:
                    if ' ' not in word:
                        try:
                            sims = self.glove.most_similar(word.lower(), topn=5)
                            vocab.update(w for w, s in sims if s > 0.65 and w.isalpha() and len(w) > 3)
                        except: pass
            patterns[topic] = list(vocab)

        matched_data = []
        cols = ['date', 'rank', 'episodeName', 'description', 'sentiment_score', 'showUri']
        
        with open(settings.SENTIMENT_DATA, 'rb') as f:
            total_rows = sum(1 for _ in f) - 1
        total_expected_chunks = math.ceil(total_rows / settings.CHUNK_SIZE)
        
        with tqdm(total=total_expected_chunks, desc="C++ Scan Progress", unit="chk", dynamic_ncols=True) as pbar:
            for chunk in pd.read_csv(settings.SENTIMENT_DATA, chunksize=settings.CHUNK_SIZE, usecols=cols, low_memory=False):
                t0 = time.time()
                
                chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
                chunk['popularity'] = (pd.to_numeric(chunk['rank'], errors='coerce').max() + 1) - pd.to_numeric(chunk['rank'], errors='coerce')
                chunk['ctx'] = (chunk['episodeName'].fillna('') + " " + chunk['description'].fillna('')).str.lower()
                
                texts_list = chunk['ctx'].tolist()
                cpp_matches = fast_scanner.scan_chunks(texts_list, patterns)
                
                if cpp_matches:
                    df_matches = pd.DataFrame(cpp_matches, columns=['row_idx', 'topic', 'matched_word'])
                    original_rows = chunk.iloc[df_matches['row_idx']].reset_index(drop=True)
                    original_rows['topic'] = df_matches['topic']
                    original_rows['matched_word'] = df_matches['matched_word']
                    matched_data.append(original_rows)

                del chunk['ctx']
                del chunk, texts_list, cpp_matches
                gc.collect()
                
                process_time = max((time.time() - t0), 0.001)
                speed = settings.CHUNK_SIZE / process_time
                ram_gb = psutil.virtual_memory().used / (1024**3)
                pbar.set_postfix({"Rows/sec": f"{speed:,.0f}", "RAM": f"{ram_gb:.1f}GB"})
                pbar.update(1)

        if not matched_data: return

        df_all = pd.concat(matched_data, ignore_index=True)
        
        df_all['sentiment_score'] = pd.to_numeric(df_all['sentiment_score'], errors='coerce')
        df_all['popularity'] = pd.to_numeric(df_all['popularity'], errors='coerce')
        
        df_topic = df_all.drop_duplicates(subset=['showUri', 'topic', 'date'])
        df_topic.groupby(['topic', 'date']).agg({'sentiment_score': 'mean', 'popularity': 'mean', 'showUri': 'count'}
        ).rename(columns={'sentiment_score':'avg_sentiment', 'popularity':'avg_popularity', 'showUri':'sample_size'}).reset_index().to_csv(settings.TOPIC_METRICS, index=False)

        df_word = df_all.drop_duplicates(subset=['showUri', 'topic', 'matched_word', 'date'])
        df_word.groupby(['topic', 'matched_word', 'date']).agg({'sentiment_score': 'mean', 'popularity': 'mean', 'showUri': 'count'}
        ).rename(columns={'sentiment_score':'avg_sentiment', 'popularity':'avg_popularity', 'showUri':'sample_size'}).reset_index().to_csv(settings.WORD_METRICS, index=False)