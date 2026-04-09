import json
import os
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from llm_survey.rag_pipeline import RAGModelExtractor
from llm_survey.topic_analysis import TopicAnalyzer


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing in environment.")

    input_file = "data/raw/synthetic_workplace_survey.csv"
    started = time.time()
    checkpoints = []

    t0 = time.time()
    extractor = RAGModelExtractor(openai_api_key=api_key, enable_literature_retrieval=False)
    checkpoints.append({"step": "init_extractor", "seconds": round(time.time() - t0, 2)})

    t0 = time.time()
    chunks = extractor.process_and_store_data(input_file, save_processed=True)
    checkpoints.append(
        {"step": "process_and_store", "seconds": round(time.time() - t0, 2), "chunks": len(chunks)}
    )

    t0 = time.time()
    extraction_results = extractor.extract_models_from_all_chunks(
        use_rag=True,
        num_context_docs=3,
        save_results=True,
    )
    successful = sum(1 for r in extraction_results if r.get("success"))
    checkpoints.append(
        {
            "step": "extract_models",
            "seconds": round(time.time() - t0, 2),
            "successful": successful,
            "total": len(extraction_results),
        }
    )

    t0 = time.time()
    topic_analyzer = TopicAnalyzer(embedding_model=extractor.embedding_model_name, nr_topics=10, min_topic_size=2)
    texts = [c["text"] for c in chunks]
    topic_results = topic_analyzer.analyze_topics(texts, save_results=True)
    checkpoints.append(
        {
            "step": "topic_analysis",
            "seconds": round(time.time() - t0, 2),
            "topics": len([t for t in topic_results["topic_info"] if t["Topic"] != -1]),
        }
    )

    report = {
        "status": "ok",
        "input_file": input_file,
        "total_runtime_seconds": round(time.time() - started, 2),
        "checkpoints": checkpoints,
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/smoke_e2e_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report))


if __name__ == "__main__":
    main()
