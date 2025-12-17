import json
from pathlib import Path
from statistics import mean
from langchain_openai import OpenAIEmbeddings
from ..rag.retriever import PgVectorRetriever
from ..mlflow_logger import start_run, end_run


def load_eval_dataset(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(k: int = 5):
    # eval_dataset.json lives alongside this script
    dataset_path = Path(__file__).resolve().parent / "eval_dataset.json"
    samples = load_eval_dataset(dataset_path)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    retriever = PgVectorRetriever(embeddings=embeddings, top_k=k)

    hits = []

    for sample in samples:
        question = sample["question"]
        expected_filename = sample["expected_doc_filename"]

        docs = retriever.get_relevant_information(question)
        retrieved_filenames = {
            d.metadata.get("file_name") for d in docs if d.metadata.get("file_name")
        }

        hit = expected_filename in retrieved_filenames
        hits.append(1.0 if hit else 0.0)

    hit_rate_at_k = mean(hits) if hits else 0.0

    start_run(
        name="rag_retrieval_eval",
        params={
            "top_k": k,
            "num_questions": len(samples),
        },
    )

    metrics = {"hit_rate_at_k": float(hit_rate_at_k)}
    artifacts = {"eval_dataset": str(dataset_path)}
    end_run(metrics=metrics, artifacts=artifacts)

    print(f"Evaluation complete. hit_rate@{k}: {hit_rate_at_k:.3f}")


if __name__ == "__main__":
    run_evaluation()
