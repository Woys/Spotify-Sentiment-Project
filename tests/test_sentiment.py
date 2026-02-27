from spotify_sentiment.pipeline.steps_sentiment import SentimentStep

def test_sentiment_normalization():
    step = SentimentStep()
    try:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
    except: pass
    
    step.execute = lambda: None
    step._normalize_score = SentimentStep._normalize_score.__get__(step)
    
    pos_score = step._normalize_score("This is absolutely wonderful and great!")
    neg_score = step._normalize_score("This is terrible, awful, and bad.")
    
    assert pos_score > 0.5
    assert neg_score < 0.5
    assert step._normalize_score("") == 0.5
