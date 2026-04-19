from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 LLM tasks: 10 summaries and one service response."
    )
    parser.add_argument(
        "--data-path",
        default=str(Path(__file__).resolve().parent / "Gift_Cards.json"),
        help="Path to the review dataset in JSON lines format.",
    )
    parser.add_argument(
        "--output-path",
        default=str(Path(__file__).resolve().parent / "artifacts" / "reports" / "llm_outputs.json"),
        help="Where to save the LLM task results JSON.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local Hugging Face model name or path.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device for transformers pipeline. Use -1 for CPU, 0 for first GPU.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for review sampling.",
    )
    return parser


def load_generator(model_name: str, device: int):
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers is not installed in the environment, so the LLM tasks cannot run."
        ) from exc

    try:
        return hf_pipeline("text-generation", model=model_name, device=device)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Could not load the Hugging Face model '{model_name}'. "
            "Make sure it is available locally or already cached."
        ) from exc


def generate_chat_text(generator, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    result = generator(messages, max_new_tokens=max_new_tokens, do_sample=False)
    generated = result[0]["generated_text"]
    if isinstance(generated, list):
        return generated[-1]["content"].strip()
    return str(generated).strip()


def run_tasks(data_path: Path, output_path: Path, model_name: str, device: int, random_state: int) -> None:
    df = pd.read_json(data_path, lines=True)
    df["reviewText"] = df["reviewText"].fillna("").astype(str)
    df["word_count"] = df["reviewText"].apply(lambda text: len(str(text).split()))

    long_reviews = df[df["word_count"] > 100].sample(n=10, random_state=random_state).reset_index(drop=True)
    question_reviews = df[
        df["reviewText"].str.contains(r"\?", na=False) &
        (df["word_count"] >= 20)
    ].copy()

    if question_reviews.empty:
        raise RuntimeError("No qualifying question-style review was found in the dataset.")

    selected_question_review = question_reviews.iloc[0]
    generator = load_generator(model_name, device)

    summaries: list[dict[str, object]] = []
    for index, row in long_reviews.iterrows():
        prompt = [{
            "role": "user",
            "content": (
                "You are an assistant. Summarise the following customer review in 50 words or fewer. "
                "Return only the summary with no extra text.\n\nReview: " + row["reviewText"]
            ),
        }]
        summary = generate_chat_text(generator, prompt, max_new_tokens=80)
        summaries.append(
            {
                "review_index": int(index),
                "word_count": int(row["word_count"]),
                "original_text": row["reviewText"],
                "summary": summary,
            }
        )

    service_prompt = [
        {
            "role": "system",
            "content": (
                "You are a professional and empathetic customer service representative "
                "for an Amazon Gift Cards product. Respond politely, address the "
                "customer's concern directly, and offer helpful next steps."
            ),
        },
        {
            "role": "user",
            "content": "Customer review: " + selected_question_review["reviewText"],
        },
    ]
    service_response = generate_chat_text(generator, service_prompt, max_new_tokens=150)

    payload = {
        "model": model_name,
        "device": device,
        "random_state": random_state,
        "summary_task": {
            "selected_reviews_count": len(summaries),
            "first_two_for_report": summaries[:2],
            "all_reviews": summaries,
        },
        "service_response_task": {
            "selected_review": {
                "overall": float(selected_question_review["overall"]),
                "word_count": int(selected_question_review["word_count"]),
                "reviewText": selected_question_review["reviewText"],
            },
            "response": service_response,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"Saved LLM task results to: {output_path}")


def main() -> None:
    args = build_parser().parse_args()
    run_tasks(
        data_path=Path(args.data_path).resolve(),
        output_path=Path(args.output_path).resolve(),
        model_name=args.model,
        device=args.device,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
