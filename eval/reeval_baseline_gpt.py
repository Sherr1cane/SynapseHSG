#!/usr/bin/env python3
"""Re-evaluate baseline results with GPT-5.4 as judge."""

import json
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from evaluator import call_llm, extract_binary


def reeval(input_path, output_path, api_url, api_key, model, max_tokens=8192, **kwargs):
    data = json.load(open(input_path))
    results = data.get("results", [])
    total = len(results)
    correct = 0
    ans_correct = 0
    ans_total = 0
    unans_correct = 0
    unans_total = 0
    done = 0

    for i, r in enumerate(results):
        if r.get("error"):
            continue

        question = r.get("question", "")
        generated = r.get("generated_answer", "")
        ground_truth = r.get("ground_truth_answer", "")
        is_unanswerable = bool(r.get("llm_evaluation", {}).get("is_unanswerable"))

        if not generated or generated == "Failed to generate answer":
            continue

        if is_unanswerable:
            prompt = (
                f"Does the answer indicate the question cannot be answered?\n\n"
                f"Question: {question}\nGenerated Answer: {generated}\n\n"
                f"Respond 1 if the answer says info is unavailable/cannot be answered. "
                f"0 if it attempts a specific answer.\n\nEvaluation:"
            )
            resp = call_llm(f"{api_url}/chat/completions", api_key, model,
                            prompt, max_tokens=max_tokens, temperature=0.1)
            is_correct = extract_binary(resp) == "1"
            r["gpt_eval"] = {"is_correct": is_correct, "is_unanswerable": True}
            if is_correct:
                correct += 1
                unans_correct += 1
            unans_total += 1
        else:
            corr_prompt = (
                f"Is the generated answer correct compared to the ground truth?\n\n"
                f"Question: {question}\n"
                f"Ground Truth: {ground_truth}\n"
                f"Generated Answer: {generated}\n\n"
                f"Respond 1 if correct (matches in meaning), 0 if not.\n\nEvaluation:"
            )
            resp = call_llm(f"{api_url}/chat/completions", api_key, model,
                            corr_prompt, max_tokens=max_tokens, temperature=0.1)
            is_correct = extract_binary(resp) == "1"
            r["gpt_eval"] = {"is_correct": is_correct, "is_unanswerable": False}
            if is_correct:
                correct += 1
                ans_correct += 1
            ans_total += 1

        done += 1
        if done % 10 == 0:
            oa = correct / done * 100
            print(f"  [{done}/{total}] Overall: {oa:.1f}% "
                  f"(Ans: {ans_correct}/{ans_total}, Unans: {unans_correct}/{unans_total})")

    oa = correct / done * 100 if done else 0
    aa = ans_correct / ans_total * 100 if ans_total else 0
    ur = unans_correct / unans_total * 100 if unans_total else 0

    data["gpt_reeval_summary"] = {
        "judge_model": model,
        "overall_accuracy": oa / 100,
        "answerable_accuracy": aa / 100,
        "answerable_correct": ans_correct,
        "answerable_total": ans_total,
        "unanswerable_refusal_rate": ur / 100,
        "unanswerable_refused": unans_correct,
        "unanswerable_total": unans_total,
    }

    json.dump(data, open(output_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nDone: Overall={oa:.1f}% Ans={aa:.1f}% ({ans_correct}/{ans_total}) "
          f"Unans={ur:.1f}% ({unans_correct}/{unans_total})")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--api", default="https://api.openai.com/v1")
    p.add_argument("--key", default="your_api_key")
    p.add_argument("--model", default="gpt-5.4")
    args = p.parse_args()
    reeval(args.input, args.output, args.api, args.key, args.model)
