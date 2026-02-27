import pandas as pd
import plotly.express as px
from spotify_sentiment.pipeline.base import PipelineStep
from spotify_sentiment.core.config import settings


class VisualizeStep(PipelineStep):
    @property
    def step_name(self) -> str:
        return "N-Weighted Dashboard Generation"

    def _calculate_weighted_aggregates(self, df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy["weighted_sentiment"] = df_copy["avg_sentiment"] * df_copy["sample_size"]
        df_copy["weighted_popularity"] = df_copy["avg_popularity"] * df_copy["sample_size"]

        agg = (
            df_copy.groupby(group_cols)
            .agg(
                {
                    "weighted_sentiment": "sum",
                    "weighted_popularity": "sum",
                    "sample_size": "sum",
                }
            )
            .reset_index()
        )

        agg["avg_sentiment"] = agg["weighted_sentiment"] / agg["sample_size"]
        agg["avg_popularity"] = agg["weighted_popularity"] / agg["sample_size"]
        return agg

    def _write_fig(self, fig, filename: str) -> None:
        """
        Key fix for "stacking Plotly" issues in a multi-HTML dashboard:

        1) Each figure HTML loads Plotly from CDN (not embedded) so you don't end up
           with repeated huge inline plotly.js blobs.
        2) Also make it responsive so it behaves inside iframes / containers.
        """
        out_path = settings.ASSETS_DIR / filename
        fig.write_html(
            out_path,
            include_plotlyjs="cdn",
            full_html=True,
            config={"responsive": True, "displaylogo": False},
        )

    def _write_index(self, html_files: list[str], filename: str = "index.html") -> None:
        """
        Optional but recommended: avoid Plotly stacking/overlap by not concatenating
        multiple full Plotly HTML docs into one page.
        Instead, render each chart in an iframe grid.
        """
        cards = "\n".join(
            f"""
            <div class="card">
              <iframe src="{f}" loading="lazy"></iframe>
              <div class="label">{f}</div>
            </div>
            """.strip()
            for f in html_files
        )

        index_html = f"""<!doctype html>
            <html lang="en">
            <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>Spotify Sentiment Dashboard</title>
            <style>
                :root {{
                --bg: #0b0b0f;
                --card: #12121a;
                --text: #e8e8f0;
                --muted: #a9a9b3;
                --border: rgba(255,255,255,0.08);
                }}
                body {{
                margin: 0;
                font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
                background: var(--bg);
                color: var(--text);
                }}
                header {{
                padding: 16px 20px;
                border-bottom: 1px solid var(--border);
                position: sticky;
                top: 0;
                background: linear-gradient(to bottom, rgba(11,11,15,0.95), rgba(11,11,15,0.80));
                backdrop-filter: blur(10px);
                z-index: 10;
                }}
                h1 {{
                margin: 0;
                font-size: 16px;
                font-weight: 650;
                letter-spacing: 0.2px;
                }}
                .wrap {{
                padding: 16px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
                gap: 14px;
                }}
                .card {{
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 14px;
                overflow: hidden;
                box-shadow: 0 8px 24px rgba(0,0,0,0.28);
                display: flex;
                flex-direction: column;
                min-height: 420px;
                }}
                iframe {{
                width: 100%;
                height: 520px;
                border: 0;
                background: #000;
                }}
                .label {{
                padding: 10px 12px;
                font-size: 12px;
                color: var(--muted);
                border-top: 1px solid var(--border);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                }}
            </style>
            </head>
            <body>
            <header><h1>Spotify Sentiment Dashboard</h1></header>
            <div class="wrap">
                {cards}
            </div>
            </body>
            </html>
        """
        (settings.ASSETS_DIR / filename).write_text(index_html, encoding="utf-8")

    def execute(self) -> None:
        settings.ASSETS_DIR.mkdir(exist_ok=True, parents=True)
        if not settings.TOPIC_METRICS.exists():
            return

        written_files: list[str] = []

        df_t = pd.read_csv(settings.TOPIC_METRICS)
        df_t["date"] = pd.to_datetime(df_t["date"])
        agg_t = self._calculate_weighted_aggregates(df_t, ["topic"])
        total_t_n = int(agg_t["sample_size"].sum())

        fig01 = px.scatter(
            agg_t,
            x="avg_sentiment",
            y="avg_popularity",
            size="sample_size",
            color="sample_size",
            text="topic",
            color_continuous_scale="Viridis",
            hover_data={"sample_size": True, "avg_sentiment": ":.3f"},
            title=f"All-Time Topic Matrix: Weighted Rank vs Sentiment (Total N={total_t_n:,})",
            template="plotly_dark",
            size_max=80,
        )
        self._write_fig(fig01, "01_global_topic_matrix_N.html")
        written_files.append("01_global_topic_matrix_N.html")

        fig02 = px.scatter(
            df_t,
            x="date",
            y="avg_popularity",
            size="sample_size",
            color="topic",
            hover_data={"sample_size": True, "avg_sentiment": True},
            title="All-Time Daily Topic Rank Trajectory (Markers Sized by Daily Sample Size N)",
            template="plotly_dark",
            size_max=50,
        )
        self._write_fig(fig02, "02_global_topic_daily_trajectory_N.html")
        written_files.append("02_global_topic_daily_trajectory_N.html")

        fig03 = px.bar(
            agg_t.sort_values("sample_size", ascending=False),
            x="topic",
            y="sample_size",
            color="sample_size",
            color_continuous_scale="Magma",
            text_auto=".2s",
            title=f"Total Confirmed Sample Size N per Topic (Total System N={total_t_n:,})",
            template="plotly_dark",
        )
        fig03.update_layout(xaxis_tickangle=-45)
        self._write_fig(fig03, "03_global_topic_absolute_volume_N.html")
        written_files.append("03_global_topic_absolute_volume_N.html")

        fig03b = px.bar(
            agg_t.sort_values("avg_sentiment", ascending=False),
            x="topic",
            y="avg_sentiment",
            color="avg_sentiment",
            color_continuous_scale="RdYlGn",
            range_color=[0, 1],
            text_auto=".3f",
            title=f"Overall Average Sentiment per Topic (Total System N={total_t_n:,})",
            template="plotly_dark",
        )
        fig03b.update_layout(xaxis_tickangle=-45)
        self._write_fig(fig03b, "03b_global_topic_sentiment_bar.html")
        written_files.append("03b_global_topic_sentiment_bar.html")

        fig04 = px.area(
            df_t.groupby(["date", "topic"])["sample_size"].sum().reset_index(),
            x="date",
            y="sample_size",
            color="topic",
            title="All-Time Daily Sample Size Volume N per Topic",
            template="plotly_dark",
        )
        self._write_fig(fig04, "04_global_topic_daily_volume_area_N.html")
        written_files.append("04_global_topic_daily_volume_area_N.html")

        fig04b = px.line(
            df_t,
            x="date",
            y="avg_sentiment",
            color="topic",
            markers=True,
            title="All-Time Daily Average Sentiment Trend per Topic",
            template="plotly_dark",
        )
        self._write_fig(fig04b, "04b_global_topic_daily_sentiment_trend.html")
        written_files.append("04b_global_topic_daily_sentiment_trend.html")

        if settings.WORD_METRICS.exists():
            df_w = pd.read_csv(settings.WORD_METRICS)
            if not df_w.empty:
                df_w["date"] = pd.to_datetime(df_w["date"])
                agg_w = self._calculate_weighted_aggregates(df_w, ["topic", "matched_word"])
                total_w_n = int(agg_w["sample_size"].sum())
                top_words = agg_w.sort_values("sample_size", ascending=False)

                fig05 = px.scatter(
                    top_words.head(75),
                    x="avg_sentiment",
                    y="avg_popularity",
                    size="sample_size",
                    color="sample_size",
                    color_continuous_scale="Plasma",
                    hover_name="matched_word",
                    text="matched_word",
                    hover_data={"sample_size": True, "topic": True},
                    title=f"All-Time Top 75 Global Keywords Matrix (Sized/Colored by N, Total Keyword N={total_w_n:,})",
                    template="plotly_dark",
                    size_max=65,
                )
                self._write_fig(fig05, "05_global_keyword_matrix_N.html")
                written_files.append("05_global_keyword_matrix_N.html")

                fig06 = px.bar(
                    top_words.head(50),
                    x="matched_word",
                    y="sample_size",
                    color="sample_size",
                    color_continuous_scale="Inferno",
                    text_auto=".2s",
                    hover_data={"topic": True},
                    title="All-Time Total Confirmed Sample Size N per Keyword (Top 50 Global)",
                    template="plotly_dark",
                )
                fig06.update_layout(xaxis_tickangle=-45)
                self._write_fig(fig06, "06_global_keyword_volume_N.html")
                written_files.append("06_global_keyword_volume_N.html")

                fig06b = px.bar(
                    top_words.head(50).sort_values("avg_sentiment", ascending=False),
                    x="matched_word",
                    y="avg_sentiment",
                    color="avg_sentiment",
                    color_continuous_scale="RdYlGn",
                    range_color=[0, 1],
                    text_auto=".3f",
                    hover_data={"topic": True},
                    title="All-Time Average Sentiment of Top 50 Global Keywords",
                    template="plotly_dark",
                )
                fig06b.update_layout(xaxis_tickangle=-45)
                self._write_fig(fig06b, "06b_global_keyword_sentiment_bar.html")
                written_files.append("06b_global_keyword_sentiment_bar.html")

                topics = agg_w["topic"].unique()
                for i, t in enumerate(topics, start=7):
                    t_safe = str(t).replace(" ", "_").lower()
                    topic_data = agg_w[agg_w["topic"] == t]
                    t_w_data = df_w[df_w["topic"] == t]

                    figA = px.scatter(
                        topic_data,
                        x="avg_sentiment",
                        y="avg_popularity",
                        size="sample_size",
                        color="sample_size",
                        color_continuous_scale="Turbo",
                        hover_name="matched_word",
                        text="matched_word",
                        hover_data={"sample_size": True},
                        title=f"[{t}] All-Time Keyword Sentiment/Rank Matrix",
                        template="plotly_dark",
                        size_max=55,
                    )
                    fnameA = f"{i:02d}a_{t_safe}_all_keywords_matrix_N.html"
                    self._write_fig(figA, fnameA)
                    written_files.append(fnameA)

                    fig_vol = px.bar(
                        topic_data.sort_values("sample_size", ascending=False),
                        x="matched_word",
                        y="sample_size",
                        color="sample_size",
                        color_continuous_scale="Turbo",
                        text_auto=".2s",
                        title=f"[{t}] All-Time Absolute Sample Size N per Keyword",
                        template="plotly_dark",
                    )
                    fig_vol.update_layout(xaxis_tickangle=-45)
                    fnameB = f"{i:02d}b_{t_safe}_all_keywords_volume_N.html"
                    self._write_fig(fig_vol, fnameB)
                    written_files.append(fnameB)

                    fig_sent = px.bar(
                        topic_data.sort_values("avg_sentiment", ascending=False),
                        x="matched_word",
                        y="avg_sentiment",
                        color="avg_sentiment",
                        color_continuous_scale="RdYlGn",
                        range_color=[0, 1],
                        text_auto=".3f",
                        title=f"[{t}] All-Time Average Sentiment per Keyword",
                        template="plotly_dark",
                    )
                    fig_sent.update_layout(xaxis_tickangle=-45)
                    fnameB2 = f"{i:02d}b2_{t_safe}_all_keywords_sentiment_bar.html"
                    self._write_fig(fig_sent, fnameB2)
                    written_files.append(fnameB2)

                    figC = px.area(
                        t_w_data.groupby(["date", "matched_word"])["sample_size"].sum().reset_index(),
                        x="date",
                        y="sample_size",
                        color="matched_word",
                        title=f"[{t}] All-Time Daily Sample Size Volume N per Keyword",
                        template="plotly_dark",
                    )
                    fnameC = f"{i:02d}c_{t_safe}_keyword_daily_volume_area_N.html"
                    self._write_fig(figC, fnameC)
                    written_files.append(fnameC)

                    top_10 = topic_data.sort_values("sample_size", ascending=False).head(10)["matched_word"]
                    figC2 = px.line(
                        t_w_data[t_w_data["matched_word"].isin(top_10)],
                        x="date",
                        y="avg_sentiment",
                        color="matched_word",
                        markers=True,
                        title=f"[{t}] All-Time Daily Sentiment Trend (Top 10 Keywords)",
                        template="plotly_dark",
                    )
                    fnameC2 = f"{i:02d}c2_{t_safe}_top_keywords_daily_sentiment_trend.html"
                    self._write_fig(figC2, fnameC2)
                    written_files.append(fnameC2)


        self._write_index(written_files, "index.html")