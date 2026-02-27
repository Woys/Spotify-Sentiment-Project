import pandas as pd
from spotify_sentiment.pipeline.steps_visualize import VisualizeStep

def test_weighted_aggregates():
    step = VisualizeStep()
    df = pd.DataFrame({
        'topic': ['AI', 'AI'],
        'avg_sentiment': [1.0, 0.0],
        'avg_popularity': [100, 50],
        'sample_size': [90, 10]
    })
    res = step._calculate_weighted_aggregates(df, ['topic'])
    assert res['avg_sentiment'].iloc[0] == 0.9
    assert res['avg_popularity'].iloc[0] == 95.0
    assert res['sample_size'].iloc[0] == 100
