#!/usr/bin/env python3
"""SFT training for Semantic Ontology Decomposer.

Trains a model to map: question → decomposition JSON (6 slots).

Base model: Qwen3.5-7B (configurable)
Framework: transformers + peft (LoRA) + trl (SFTTrainer)

Usage:
    # Single GPU
    python train/train_decomposer.py \
        --train-data train/decomposer_train.jsonl \
        --dev-data train/decomposer_dev.jsonl \
        --output-dir models/decomposer_qwen35_9b \
        --base-model Qwen/Qwen3.5-9B \
        --epochs 3 --lr 2e-4 --batch-size 4

    # Multi-GPU with accelerate
    accelerate launch train/train_decomposer.py \
        --train-data train/decomposer_train.jsonl \
        --dev-data train/decomposer_dev.jsonl \
        --output-dir models/decomposer_qwen35_9b \
        --base-model Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = """You are a semantic decomposition model for academic paper QA. Given a question about a research paper, output a structured JSON decomposition with these fields:
- target_semantic_unit: one of [Claim, Method, Result, Metric, Limitation]
- target_relation: one of [supported_by, emphasizes, compares, measured_by, aligned_to_slide, grounded_in_paper, limitation_of, improvement_over, other]
- constraints: object with keys [time_constraint, page_constraint, baseline_constraint, metric_constraint, section_constraint, experiment_setting], values null or string
- modality_requirement: one of [paper, audio, visual, paper+audio, paper+visual, audio+visual, paper+audio+visual, none]
- reasoning_pattern: one of [single_hop, evidence_linking, cross_slide, cross_time, comparison, limitation_or_caveat, multi_hop_aggregation, unanswerable]
- answer_type: one of [entity, phrase, number, explanation, yes_no, abstain]
- answerable: boolean

Output ONLY valid JSON, no markdown, no explanation."""


def load_sft_data(path: str, validate: bool = True) -> List[Dict[str, Any]]:
    """Load JSONL and convert to SFT messages format."""
    samples = []
    invalid_count = 0
    validation_errors = []

    for line_num, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue

        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            invalid_count += 1
            validation_errors.append(f"Line {line_num}: JSON parse error - {e}")
            continue

        decomp = raw.get("decomposition", {})
        question = raw.get("question", "")

        if not question or not decomp:
            invalid_count += 1
            validation_errors.append(f"Line {line_num}: Missing question or decomposition")
            continue

        # Validate decomposition structure if requested
        if validate:
            required_slots = ["target_semantic_unit", "target_relation", "constraints",
                            "modality_requirement", "reasoning_pattern", "answer_type", "answerable"]
            missing_slots = [slot for slot in required_slots if slot not in decomp]
            if missing_slots:
                invalid_count += 1
                validation_errors.append(f"Line {line_num}: Missing slots: {missing_slots}")
                continue

            # Validate constraints object
            if not isinstance(decomp.get("constraints"), dict):
                invalid_count += 1
                validation_errors.append(f"Line {line_num}: constraints must be an object")
                continue

        # Serialize decomposition to clean JSON
        decomp_json = json.dumps(decomp, ensure_ascii=False)

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": decomp_json},
            ],
            "sample_id": raw.get("sample_id", ""),
        })

    if invalid_count > 0:
        print(f"Warning: {invalid_count} invalid samples found in {path}")
        if invalid_count <= 10:
            for error in validation_errors:
                print(f"  {error}")
        else:
            print(f"  First 10 errors:")
            for error in validation_errors[:10]:
                print(f"  {error}")

    return samples


def compute_metrics(predictions: List[str], references: List[Dict]) -> Dict[str, Any]:
    """Compute decomposer evaluation metrics."""
    import json as _json

    n = len(predictions)
    n_valid = 0
    slot_correct = {
        "target_semantic_unit": 0,
        "target_relation": 0,
        "modality_requirement": 0,
        "reasoning_pattern": 0,
        "answer_type": 0,
        "answerable": 0,
    }
    n_exact_match = 0
    constraint_matches = 0
    n_constraint_samples = 0

    for pred_str, ref in zip(predictions, references):
        ref_decomp = ref.get("decomposition", {})

        # Parse prediction
        try:
            pred_decomp = _json.loads(pred_str.strip())
            n_valid += 1
        except _json.JSONDecodeError:
            # Try extracting JSON
            start = pred_str.find("{")
            end = pred_str.rfind("}")
            if start >= 0 and end > start:
                try:
                    pred_decomp = _json.loads(pred_str[start:end + 1])
                    n_valid += 1
                except _json.JSONDecodeError:
                    continue
            else:
                continue

        # Per-slot accuracy
        all_match = True
        for slot in slot_correct:
            if pred_decomp.get(slot) == ref_decomp.get(slot):
                slot_correct[slot] += 1
            else:
                all_match = False

        if all_match:
            n_exact_match += 1

        # Constraint accuracy
        ref_constraints = ref_decomp.get("constraints", {})
        pred_constraints = pred_decomp.get("constraints", {})
        non_null_keys = [k for k, v in ref_constraints.items() if v is not None and v != "null"]
        if non_null_keys:
            n_constraint_samples += 1
            if all(pred_constraints.get(k) == ref_constraints.get(k) for k in non_null_keys):
                constraint_matches += 1

    return {
        "n_samples": n,
        "json_valid_rate": n_valid / max(1, n),
        "exact_match_rate": n_exact_match / max(1, n),
        "slot_accuracy": {k: v / max(1, n) for k, v in slot_correct.items()},
        "constraint_accuracy": constraint_matches / max(1, n_constraint_samples),
        "n_constraint_samples": n_constraint_samples,
    }


def train(args: argparse.Namespace) -> None:
    """Main training function."""
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer

    # Load data
    print("Loading training data...")
    train_samples = load_sft_data(args.train_data)
    dev_samples = load_sft_data(args.dev_data) if args.dev_data else []
    print(f"Train: {len(train_samples)}, Dev: {len(dev_samples)}")

    # Convert to HF dataset
    def to_dataset(samples):
        return Dataset.from_list([
            {"messages": json.dumps(s["messages"], ensure_ascii=False)}
            for s in samples
        ])

    train_ds = to_dataset(train_samples)
    dev_ds = to_dataset(dev_samples) if dev_samples else None

    # Load tokenizer and model — use local path to avoid HuggingFace hub calls
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    model_path = args.base_model
    if not os.path.isdir(model_path):
        # Try cached HuggingFace model
        cache_snap = os.path.expanduser(
            f"~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/"
        )
        if os.path.isdir(cache_snap):
            snaps = os.listdir(cache_snap)
            if snaps:
                model_path = os.path.join(cache_snap, snaps[0])
                print(f"Using cached model: {model_path}")
    print(f"Loading base model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect bf16/fp16 support
    import torch
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    # Calculate warmup steps based on 5% of total training steps
    total_steps = args.epochs * (len(train_samples) // (args.batch_size * args.grad_accum))
    warmup_steps = max(1, int(0.05 * total_steps))

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,  # Use warmup_steps instead of warmup_ratio
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if dev_ds else "no",
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        save_total_limit=3,
        metric_for_best_model="eval_loss" if dev_ds else None,
        load_best_model_at_end=True if dev_ds else False,
        # Memory optimization
        max_grad_norm=1.0,
        optim="adamw_torch_fused",  # More memory efficient
        dataloader_num_workers=2,  # Reduce CPU memory
        per_device_eval_batch_size=1,  # Reduce eval batch size to avoid OOM
        # Additional memory saving
        torch_compile=False,  # Disable compilation to save memory
        ddp_find_unused_parameters=False,
        # Evaluation settings to prevent OOM
        eval_accumulation_steps=1,  # Don't accumulate predictions in memory
        prediction_loss_only=True,  # Only compute loss, don't store predictions
    )

    # Data collation: format messages as chat
    def format_messages(example):
        messages = json.loads(example["messages"])
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = train_ds.map(format_messages)
    if dev_ds:
        dev_ds = dev_ds.map(format_messages)

    # Custom compute_metrics function for evaluation
    def compute_metrics_wrapper(eval_preds):
        """Wrapper to compute metrics during evaluation."""
        logits, labels = eval_preds
        # For SFT, we typically use perplexity/loss, so we skip complex metrics here
        # You can add more sophisticated metrics if needed
        return {}

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_wrapper if dev_ds else None,
    )

    print("Starting training...")
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Train samples: {len(train_samples)}")
    if dev_ds:
        print(f"Dev samples: {len(dev_samples)}")

    trainer.train()

    # Save
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save training config
    config = {
        "base_model": args.base_model,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "train_samples": len(train_samples),
        "dev_samples": len(dev_samples),
        "bf16": use_bf16,
        "fp16": use_fp16,
        "target_modules": args.target_modules,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Save training summary
    summary = {
        "status": "completed",
        "final_dir": str(final_dir),
        "train_samples": len(train_samples),
        "dev_samples": len(dev_samples),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nTraining complete. Model saved to {final_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT Decomposer training")
    parser.add_argument("--train-data", required=True, help="Path to training data JSONL")
    parser.add_argument("--dev-data", default="", help="Path to development data JSONL")
    parser.add_argument("--output-dir", required=True, help="Path to output directory")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size (reduce if OOM)")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (increase to maintain effective batch size)")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (reduce to save memory)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj",
                       help="Target modules for LoRA")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--no-validate", action="store_true",
                       help="Disable data validation during loading")
    args = parser.parse_args()

    # Validate arguments
    if args.epochs <= 0:
        print("Error: epochs must be positive")
        sys.exit(1)
    if args.lr <= 0:
        print("Error: learning rate must be positive")
        sys.exit(1)
    if args.batch_size <= 0:
        print("Error: batch size must be positive")
        sys.exit(1)

    # Check if train data exists
    if not Path(args.train_data).exists():
        print(f"Error: Training data not found: {args.train_data}")
        sys.exit(1)

    if args.dev_data and not Path(args.dev_data).exists():
        print(f"Warning: Dev data not found: {args.dev_data}")
        args.dev_data = ""

    try:
        train(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
