import fast_scanner

def test_scan_chunks_basic():
    texts = ["i love machine learning", "the economy is bad", "random text"]
    topic_words = {
        "AI": ["machine learning", "ai"],
        "Economy": ["economy", "inflation"]
    }
    results = fast_scanner.scan_chunks(texts, topic_words)
    assert len(results) == 2
    assert (0, "AI", "machine learning") in results
    assert (1, "Economy", "economy") in results

def test_scan_chunks_boundaries():
    texts = ["this is an ailing patient", "daily mail", "we love ai"] 
    topic_words = {"AI": ["ai"]}
    results = fast_scanner.scan_chunks(texts, topic_words)
    assert len(results) == 1
    assert (2, "AI", "ai") in results
