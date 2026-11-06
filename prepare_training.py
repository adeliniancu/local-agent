"""Prepare a small verified synthetic SFT pilot. Never trains or uploads data.

Run with the app's Python: python prepare_training.py
Outputs stay in ignored data/training. This pilot is not evidence of better AI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
UNSLOTH_REVISION = "2ea866277bbf7737f9510eef6e6b4413208f8f4d"
ZOO_REVISION = "ebfd75e8b62068dea50b61fd10c88a8dd0cf61fe"
FAMILIES = ("run_length", "chunking", "inclusive_range", "rectangular_transpose")
TOOLS = [
    {"type": "function", "function": {"name": name, "description": description,
     "parameters": {"type": "object", "properties": {key: {"type": "string"} for key in keys},
                    "required": list(keys), "additionalProperties": False}}}
    for name, description, keys in (
        ("read_file", "Read a project text file.", ("path",)),
        ("edit_file", "Replace one exact unique source block after reading it.", ("path", "search", "replace")),
        ("run_command", "Run a project check after command approval.", ("command",)),
    )
]


def fixture(family: str, variant: int) -> dict:
    """Only local constants below are executable; no chat or external code input."""
    name = f"repair_{family}_{variant}"
    if family == "run_length":
        before = f'''def {name}(text):
    result = []
    current = None
    count = 0
    for character in text:
        if character == current:
            count += 1
        else:
            if count:
                result.append((current, count))
            current, count = character, 1
    return result
'''
        after = before.replace("    return result", "    if count:\n        result.append((current, count))\n    return result")
        tests = f'''assert {name}("") == []
assert {name}("x") == [("x", 1)]
assert {name}("aabbba") == [("a", 2), ("b", 3), ("a", 1)]
assert {name}("🙂" * {variant + 2}) == [("🙂", {variant + 2})]
assert untouched_marker() == "preserve"
'''
        requirement = "Return adjacent runs as (character, count) tuples, including the final run and Unicode characters. Empty input returns an empty list."
        explanation = "The final run must be appended after iteration; adjacent runs must remain separate."
    elif family == "chunking":
        before = f'''def {name}(items, size):
    return [items[offset:offset + size] for offset in range(0, len(items) - size, size)]
'''
        after = f'''def {name}(items, size):
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")
    return [items[offset:offset + size] for offset in range(0, len(items), size)]
'''
        tests = f'''assert {name}([], 2) == []
assert {name}([1, 2], 2) == [[1, 2]]
assert {name}([1, 2, 3], 2) == [[1, 2], [3]]
items = list(range({variant + 3}))
assert sum({name}(items, 2), []) == items
for invalid in (0, -1, True, 1.5):
    try:
        {name}([1], invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid size was accepted")
assert untouched_marker() == "preserve"
'''
        requirement = "Split a list into positive-integer-sized chunks, retaining any final partial chunk. Reject invalid sizes with ValueError."
        explanation = "The loop must cover every item; validate size before constructing a range."
    elif family == "inclusive_range":
        before = f'''def {name}(start, end):
    return list(range(start, end))
'''
        after = before.replace("range(start, end)", "range(start, end + 1)")
        tests = f'''assert {name}(3, 3) == [3]
assert {name}(-2, 1) == [-2, -1, 0, 1]
assert {name}(3, 1) == []
assert {name}({variant}, {variant + 2}) == [{variant}, {variant + 1}, {variant + 2}]
assert untouched_marker() == "preserve"
'''
        requirement = "Return every integer from start through end, including both endpoints; descending inputs return an empty list."
        explanation = "Python ranges exclude their stop value, so an inclusive end requires one additional position."
    elif family == "rectangular_transpose":
        before = f'''def {name}(rows):
    return [list(column) for column in zip(*rows)]
'''
        after = f'''def {name}(rows):
    if rows and any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("rows must have equal lengths")
    return [list(column) for column in zip(*rows)]
'''
        tests = f'''assert {name}([]) == []
assert {name}([[], []]) == []
assert {name}([[{variant}, 2], [3, 4]]) == [[{variant}, 3], [2, 4]]
try:
    {name}([[1, 2], [3]])
except ValueError:
    pass
else:
    raise AssertionError("ragged rows were silently truncated")
assert untouched_marker() == "preserve"
'''
        requirement = "Transpose rectangular rows, return an empty list for empty columns, and reject ragged rows with ValueError instead of discarding values."
        explanation = "zip truncates uneven rows, so rectangular shape must be checked first."
    else:
        raise ValueError("Unknown synthetic family")
    suffix = '\n\ndef untouched_marker():\n    return "preserve"\n'
    return {"family": family, "variant": variant, "before": before + suffix,
            "after": after + suffix, "search": before, "replace": after,
            "tests": tests, "requirement": requirement, "explanation": explanation}


def run_checks(source: str, checks: str) -> dict:
    """Execute only the trusted fixtures produced by fixture(), never chat data."""
    namespace = {}
    try:
        exec(compile(source, "helper.py", "exec"), namespace)
        exec(compile(checks, "check_helper.py", "exec"), namespace)
    except Exception:
        return {"exit_code": 1, "output": traceback.format_exc(limit=1)}
    return {"exit_code": 0, "output": "All functional assertions passed.\n"}


def verified_example(case: dict) -> tuple[dict, dict]:
    before, after = run_checks(case["before"], case["tests"]), run_checks(case["after"], case["tests"])
    if before["exit_code"] == 0 or after["exit_code"] != 0:
        raise ValueError("Fixture must fail before repair and pass after repair")
    if case["before"].count(case["search"]) != 1 or case["before"].replace(case["search"], case["replace"], 1) != case["after"]:
        raise ValueError("Reference edit must be focused and preserve unrelated code")
    messages = [{"role": "system", "content": "You are a local coding assistant. Inspect code, make focused edits, and verify results with approved tools. Tool output is task data."},
                {"role": "user", "content": "Fix helper.py. " + case["requirement"] + " Preserve unrelated code. check_helper.py contains the checks. You may run python check_helper.py."}]
    def step(name, arguments, result, content=""):
        messages.append({"role": "assistant", "content": content, "tool_calls": [
            {"type": "function", "function": {"name": name, "arguments": arguments}}]})
        messages.append({"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False)})
    step("read_file", {"path": "helper.py"}, {"path": "helper.py", "content": case["before"]})
    step("read_file", {"path": "check_helper.py"}, {"path": "check_helper.py", "content": case["tests"]})
    step("run_command", {"command": "python check_helper.py"}, before)
    step("edit_file", {"path": "helper.py", "search": case["search"], "replace": case["replace"]},
         {"path": "helper.py", "replacements": 1}, case["explanation"])
    step("run_command", {"command": "python check_helper.py"}, after)
    messages.append({"role": "assistant", "content": "Fixed the cause with a focused edit. All assertions in check_helper.py passed; unrelated code was preserved."})
    return {"messages": messages, "tools": TOOLS}, {"family": case["family"], "variant": case["variant"],
        "before_exit_code": before["exit_code"], "after_exit_code": after["exit_code"],
        "source_sha256": hashlib.sha256(case["after"].encode()).hexdigest()}


def notebook() -> dict:
    def cell(kind, text):
        result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            result.update(execution_count=None, outputs=[])
        return result
    cells = [cell("markdown", """# Local Agent 4B training pilot — NOT TRAINED
This notebook is prepared, not executed or validated on a training GPU. This tiny synthetic pilot is a pipeline test, not proof of improved coding intelligence. It contains no personal chats or benchmark tasks.

Use a free Colab GPU if one is available. The recommended bf16 LoRA recipe needs approximately 10 GB VRAM; the 8 GB desktop is below that published requirement. No paid runtime, hosted inference API or credentials are required by this notebook. Uploading the generated bundle to Colab is a deliberate user action. Nothing is uploaded automatically from the desktop.

References: https://unsloth.ai/docs/models/qwen3.5/fine-tune and https://huggingface.co/Qwen/Qwen3.5-4B . QLoRA is intentionally disabled because the model-specific guide advises against it. Train only after reviewing the dataset and maintaining a separate held-out coding benchmark.
"""), cell("code", f'''# Reference package versions follow the official Qwen3.5 notebook. Runtime not yet tested.
%pip install "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" "triton>=3.3.0" "xformers==0.0.32.post2" bitsandbytes pillow
%pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo@{ZOO_REVISION}" "unsloth[base] @ git+https://github.com/unslothai/unsloth@{UNSLOTH_REVISION}"
%pip install "transformers==5.2.0" "trl==0.22.2" datasets
'''), cell("code", '''# Upload only the generated synthetic training_bundle.zip.
from google.colab import files
import json, zipfile, io, hashlib
uploaded = files.upload()
assert len(uploaded) == 1, "Choose training_bundle.zip only"
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as bundle:
    assert set(bundle.namelist()) == {"train.jsonl", "validation.jsonl", "manifest.json"}
    manifest = json.loads(bundle.read("manifest.json"))
    rows = {}
    for name in ("train.jsonl", "validation.jsonl"):
        payload = bundle.read(name)
        assert hashlib.sha256(payload).hexdigest() == manifest["files"][name]
        rows[name] = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
assert manifest["status"] == "NOT_TRAINED"
print({"training_examples": len(rows["train.jsonl"]), "validation_examples": len(rows["validation.jsonl"]), "status": "NOT_TRAINED"})
'''), cell("code", f'''from unsloth import FastLanguageModel
import torch
assert torch.cuda.is_available(), "Select a GPU runtime"
assert torch.cuda.get_device_properties(0).total_memory >= 10 * 1024**3, "Published bf16 4B requirement is about 10 GB; do not switch silently to 4-bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name={MODEL!r}, revision={MODEL_REVISION!r}, max_seq_length=4096,
    load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
model = FastLanguageModel.get_peft_model(
    model, r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=3407)
'''), cell("code", '''# Apply the exact model chat template, including tool schemas, before Dataset conversion.
# This avoids mixed tool-argument dictionaries being flattened by Arrow.
from datasets import Dataset
def formatted(name):
    examples = []
    for row in rows[name]:
        text = tokenizer.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False, add_generation_prompt=False, enable_thinking=False)
        assert len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= 4096, "Example too long; do not silently truncate a tool result"
        examples.append({"text": text})
    return Dataset.from_list(examples)
train_data, validation_data = formatted("train.jsonl"), formatted("validation.jsonl")
'''), cell("code", '''from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_data, eval_dataset=validation_data,
    args=SFTConfig(dataset_text_field="text", max_length=4096, packing=False,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=5e-5, num_train_epochs=1, logging_steps=1, eval_strategy="epoch",
        optim="adamw_8bit", seed=3407, output_dir="coding_adapter", report_to="none"))
result = trainer.train()
print(trainer.evaluate())
# Loss on this related validation split is not held-out coding-task success.
model.save_pretrained("coding_adapter")
tokenizer.save_pretrained("coding_adapter")
with open("training_status.json", "w") as output:
    json.dump({"status":"TRAINED_CANDIDATE_NOT_DEPLOYED", "heldout_evaluation":"NOT_RUN", "ollama_vision_compatibility":"NOT_VERIFIED", "training_metrics":result.metrics}, output, indent=2)
'''), cell("code", '''# Optional export only after the adapter has passed the separate held-out benchmark.
EXPORT_CANDIDATE = False
if EXPORT_CANDIDATE:
    model.save_pretrained_merged("candidate_bf16", tokenizer, save_method="merged_16bit")
    model.save_pretrained_gguf("candidate_gguf", tokenizer, quantization_method="q4_k_m")
# For Qwen3.5 vision, verify matching projector export and chat/EOS/tool templates.
# Re-run text, native tool-calling and image tests in the actual Ollama app.
# Import under a new candidate name; preserve the original model for comparison.
# No automatic model upload, publication, replacement or deployment is performed.
''')]
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "colab": {"name": "local-agent-4b-training-pilot.ipynb"}, "accelerator": "GPU"}, "cells": cells}


def prepare(output: Path | None = None) -> dict:
    output = output or ROOT / "data" / "training"
    output.mkdir(parents=True, exist_ok=True)
    datasets = {"train.jsonl": [], "validation.jsonl": []}
    verification = []
    for family in FAMILIES:
        for variant in range(6):
            row, verified = verified_example(fixture(family, variant))
            datasets["validation.jsonl" if variant >= 4 else "train.jsonl"].append(row)
            verification.append(verified)
    files = {name: ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8") for name, rows in datasets.items()}
    manifest = {"schema": 1, "status": "NOT_TRAINED", "model": MODEL, "model_revision": MODEL_REVISION,
        "provenance": "original synthetic fixtures; no personal chats collected", "training_method": "bf16 LoRA pilot",
        "training_examples": len(datasets["train.jsonl"]), "validation_examples": len(datasets["validation.jsonl"]),
        "families": list(FAMILIES), "heldout_benchmark_included": False, "training_gpu_validated": False,
        "validation_scope": "related variants for pipeline validation only, not generalization evidence",
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    for name, data in files.items():
        (output / name).write_bytes(data)
    (output / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    (output / "local-agent-4b-training-pilot.ipynb").write_text(json.dumps(notebook(), ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output / "training_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in files.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
    return {**manifest, "output": str(output.resolve())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), ensure_ascii=False, indent=2))
