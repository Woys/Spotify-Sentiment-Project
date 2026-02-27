# Spotify Podcast Sentiment Pipeline

## Description
This project is an enterprise-grade, C++ accelerated Natural Language Processing (NLP) data pipeline designed to extract market intelligence from massive datasets of daily updated Spotify podcasts. It ingests raw podcast metadata, scores sentiment using chunked VADER analysis, and maps textual content to predefined analytical topics using semantic keyword expansion (GloVe embeddings) and an ultra-fast, zero-copy custom C++ text scanner. Finally, it generates a comprehensive suite of interactive, N-weighted HTML dashboards using Plotly.

## How to Use It

Run via Docker Compose (Recommended)
The easiest way to execute the pipeline end-to-end is using Docker Compose. This ensures all C++ dependencies and Python environments are correctly sandboxed.
Bash

docker compose up --build

This will automatically map the /data and /assets volumes to your local machine.

4. Run via CLI Locally (Alternative)
If you prefer running it locally, you need Python 3.10+ and a C++ compiler.
Bash

pip install uv
uv pip install pybind11
uv pip install -e .
spotify-pipeline --step all

Note: You can run individual steps using --step [download|sentiment|analyze|visualize].

** View Dashboards
Once the pipeline finishes, open the assets/ directory in your web browser. You will find interactive Plotly HTML files containing your N-weighted visualizations.

##  Overview

This project is a high-performance, containerized analytics engine built to extract actionable market intelligence and sentiment signals from massive datasets of daily Spotify podcasts. By combining Python's rich data ecosystem with a custom-compiled C++ text scanner, the pipeline efficiently processes millions of rows to track public sentiment trends across specific economic and technological topics.
Pipeline Architecture: How It Works

The system executes a sequential 4-step pipeline, orchestrated by a central PipelineRunner that logs RAM telemetry and execution speed at every stage.
1. Data Ingestion (steps_download.py)

    Action: Connects to the Kaggle API.

    Process: Authenticates using your .env credentials and streams the daily-updated "Top Spotify Podcasts" dataset directly into the local data/ directory.

    Optimization: Checks for existing raw data to bypass redundant downloads, saving bandwidth and time during iterative testing.

2. Sentiment Analysis (steps_sentiment.py)

    Action: Scores the emotional tone of every podcast description.

    Process: Uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner). To prevent out-of-memory (OOM) crashes on large datasets, it reads the CSV in strict chunks of 50,000 rows.

    Optimization: It utilizes a dynamic cache to remember the sentiment score of unique text strings. If a podcast description repeats across days, the math is entirely skipped. Scores are normalized to a clean 0.0 (negative) to 1.0 (positive) scale.

3. C++ Hash Extraction (steps_analyze.py & fast_scanner.cpp)

    Action: Maps the raw text to specific analytical topics (e.g., parsing for financial market indicators or AI trends).

    Process:

        Semantic Expansion: Before scanning, Python uses GloVe word embeddings to dynamically expand your base keywords. If you define "economy", the model will automatically include highly correlated vector terms like "inflation" or "markets".

        Zero-Copy Scanning: The text chunks are passed into a custom-built C++ extension (fast_scanner.so). This C++ module uses an O(N) algorithm to scan strings for n-gram matches almost instantly. It bypasses Python's Global Interpreter Lock (GIL) and avoids copying memory back and forth.

    Output: Generates highly compressed aggregation tables (topic_metrics.csv and word_metrics.csv) that group the sentiment, popularity rank, and occurrence counts (N) of each keyword.

4. N-Weighted Visualization (steps_visualize.py)

    Action: Generates interactive HTML dashboards.

    Process: Reads the aggregated metrics and calculates true weighted averages based on the sample size (N) of each keyword. This mathematically prevents Simpson's Paradox—ensuring that a wildly positive sentiment score from a single obscure podcast doesn't outweigh a mildly positive score backed by 10,000 top-ranked podcasts.

    Output: Dumps a suite of interactive Plotly HTML files into the assets/ directory. This ranges from global macro-level scatter matrices to micro-level, keyword-specific daily volume charts.
