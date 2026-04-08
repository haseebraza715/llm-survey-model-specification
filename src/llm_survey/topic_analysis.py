import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import yaml
from bertopic import BERTopic
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


class TopicAnalyzer:
    """BERTopic-based thematic analysis for qualitative data."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        nr_topics: int = 10,
        min_topic_size: int = 5,
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.nr_topics = nr_topics
        self.min_topic_size = min_topic_size
        self.topic_model = None
        self.keybert_model = None

    def fit_topic_model(self, texts: List[str], save_model: bool = True) -> Tuple[List[int], np.ndarray]:
        """Fit BERTopic model to the texts."""
        print(f"Fitting BERTopic model to {len(texts)} texts...")
        self.topic_model = BERTopic(nr_topics=self.nr_topics, min_topic_size=self.min_topic_size, verbose=True)
        topics, probs = self.topic_model.fit_transform(texts)

        if save_model:
            model_path = "outputs/bertopic_model"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.topic_model.save(model_path)
            print(f"Topic model saved to {model_path}")

        return topics, probs

    def extract_keywords(self, texts: List[str], top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Extract keywords using KeyBERT."""
        print("Extracting keywords using KeyBERT...")
        self.keybert_model = KeyBERT(model=self.embedding_model)

        keywords = {}
        for i, text in enumerate(texts):
            doc_keywords = self.keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                use_maxsum=True,
                nr_candidates=20,
                top_n=top_k,
            )
            keywords[f"doc_{i}"] = doc_keywords

        return keywords

    def analyze_topics(self, texts: List[str], save_results: bool = True) -> Dict[str, Any]:
        """Perform comprehensive topic analysis."""
        topics, probs = self.fit_topic_model(texts, save_model=save_results)
        keywords = self.extract_keywords(texts)
        topic_info = self.topic_model.get_topic_info()

        topic_keywords = {}
        for topic_id in topic_info["Topic"].unique():
            if topic_id != -1:
                topic_keywords[topic_id] = self.topic_model.get_topic(topic_id)

        results = {
            "topic_info": topic_info.to_dict("records"),
            "topic_keywords": topic_keywords,
            "document_keywords": keywords,
            "topics": topics.tolist(),
            "probabilities": probs.tolist() if probs is not None else None,
            "model_info": {
                "nr_topics": self.nr_topics,
                "min_topic_size": self.min_topic_size,
                "total_documents": len(texts),
            },
        }

        if save_results:
            output_path = "outputs/topic_analysis.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Topic analysis results saved to {output_path}")

        return results

    def create_topic_visualizations(self, results: Dict[str, Any], save_plots: bool = True) -> Dict[str, str]:
        """Create visualizations for topic analysis."""
        topic_info = pd.DataFrame(results["topic_info"])
        fig_dist = px.bar(
            topic_info[topic_info["Topic"] != -1],
            x="Topic",
            y="Count",
            title="Topic Distribution",
            labels={"Count": "Number of Documents", "Topic": "Topic ID"},
        )
        fig_size = px.pie(
            topic_info[topic_info["Topic"] != -1],
            values="Count",
            names="Topic",
            title="Topic Size Distribution",
        )

        plot_paths = {}
        if save_plots:
            plots_dir = "outputs/plots"
            os.makedirs(plots_dir, exist_ok=True)
            dist_path = f"{plots_dir}/topic_distribution.html"
            fig_dist.write_html(dist_path)
            plot_paths["topic_distribution"] = dist_path
            size_path = f"{plots_dir}/topic_size_distribution.html"
            fig_size.write_html(size_path)
            plot_paths["topic_size_distribution"] = size_path
            print(f"Plots saved to {plots_dir}")

        return plot_paths

    def generate_topic_summary(self, results: Dict[str, Any], save_summary: bool = True) -> str:
        """Generate a human-readable summary of the topic analysis."""
        topic_info = pd.DataFrame(results["topic_info"])
        valid_topics = topic_info[topic_info["Topic"] != -1]

        summary = f"""# Topic Analysis Summary

## Overview
- Total documents analyzed: {results['model_info']['total_documents']}
- Number of topics identified: {len(valid_topics)}
- Minimum topic size: {results['model_info']['min_topic_size']}

## Topic Breakdown

"""

        for _, topic in valid_topics.iterrows():
            topic_id = topic["Topic"]
            count = topic["Count"]
            name = topic["Name"]
            keywords = results["topic_keywords"].get(topic_id, [])
            keyword_str = ", ".join([kw[0] for kw in keywords[:5]])
            summary += f"""### Topic {topic_id}: {name}
- **Size**: {count} documents
- **Top Keywords**: {keyword_str}
- **Percentage**: {(count / results['model_info']['total_documents'] * 100):.1f}%

"""

        noise_topic = topic_info[topic_info["Topic"] == -1]
        if not noise_topic.empty:
            noise_count = noise_topic.iloc[0]["Count"]
            summary += f"""### Noise/Outliers
- **Size**: {noise_count} documents
- **Percentage**: {(noise_count / results['model_info']['total_documents'] * 100):.1f}%

"""

        if save_summary:
            output_path = "outputs/topic_summary.md"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"Topic summary saved to {output_path}")

        return summary

    def export_topic_data(self, results: Dict[str, Any], output_format: str = "yaml") -> str:
        """Export topic analysis results in structured format."""
        export_data = {
            "metadata": {
                "total_documents": results["model_info"]["total_documents"],
                "number_of_topics": len([t for t in results["topic_info"] if t["Topic"] != -1]),
                "min_topic_size": results["model_info"]["min_topic_size"],
            },
            "topics": [],
        }

        for topic_info in results["topic_info"]:
            if topic_info["Topic"] != -1:
                topic_data = {
                    "id": topic_info["Topic"],
                    "name": topic_info["Name"],
                    "count": topic_info["Count"],
                    "keywords": [kw[0] for kw in results["topic_keywords"].get(topic_info["Topic"], [])],
                }
                export_data["topics"].append(topic_data)

        if output_format.lower() == "yaml":
            output = yaml.dump(export_data, default_flow_style=False, sort_keys=False)
            file_ext = "yaml"
        else:
            output = json.dumps(export_data, indent=2, ensure_ascii=False)
            file_ext = "json"

        output_path = f"outputs/topic_analysis.{file_ext}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)

        print(f"Topic analysis exported to {output_path}")
        return output_path

