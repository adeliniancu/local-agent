"""Development-only local model compatibility probes. Does not execute model tools."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import struct
import time
import zlib

import runtime

ROOT = Path(__file__).resolve().parent
NONCE = "adapter-probe-42"


def probe_png() -> bytes:
    """A large red left half and blue right half; no fonts or external images."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    width, height = 256, 128
    row = b"\x00" + b"\xff\x00\x00" * (width // 2) + b"\x00\x00\xff" * (width // 2)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * height)) + chunk(b"IEND", b""))


def check_response(kind: str, response: dict) -> bool:
    if response.get("done") is not True or response.get("done_reason") == "length" or response.get("error"):
        return False
    message = response.get("message", {})
    if kind == "text":
        return message.get("content", "").strip() == "READY"
    if kind == "native_tools":
        calls = message.get("tool_calls", [])
        if len(calls) != 1:
            return False
        function = calls[0].get("function", {})
        args = function.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return False
        return function.get("name") == "echo_probe" and args == {"value": NONCE}
    if kind == "vision":
        try:
            answer = json.loads(message.get("content", ""))
            return answer == {"left": "red", "right": "blue"}
        except (ValueError, TypeError):
            return False
    raise ValueError("Unknown compatibility probe")


def smoke_model(model: str) -> dict:
    info = runtime.get_model_info(model)
    options = {**runtime.get_inference_options(model), "seed": 42, "temperature": 0, "num_predict": 256}
    capabilities = info.get("capabilities", [])
    result = {"model": model, "options": options, "capabilities": capabilities,
              "advertised_tools_and_vision": "tools" in capabilities and "vision" in capabilities,
              "probes": []}
    tool = {"type": "function", "function": {"name": "echo_probe",
            "description": "Return the supplied probe value. This is a harmless compatibility probe.",
            "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                           "required": ["value"], "additionalProperties": False}}}
    requests = [
        ("text", {"messages": [{"role": "user", "content": "Reply with exactly READY and nothing else."}]}),
        ("native_tools", {"messages": [{"role": "user", "content":
            f"Call echo_probe exactly once with value {NONCE}. Do not write prose."}], "tools": [tool]}),
        ("vision", {"messages": [{"role": "user", "content":
            "Identify the dominant color of the left and right halves of this image. Use lowercase color names.",
            "images": [base64.b64encode(probe_png()).decode("ascii")]}],
            "format": {"type": "object", "properties": {"left": {"type": "string"},
                "right": {"type": "string"}}, "required": ["left", "right"], "additionalProperties": False}}),
    ]
    for kind, payload in requests:
        start = time.monotonic()
        try:
            response = runtime._post("/api/chat", {"model": model, "stream": False, "think": False,
                "keep_alive": "2m", "options": options, **payload}, timeout=300)
            passed = check_response(kind, response)
            record = {"kind": kind, "passed": passed, "response": response}
        except Exception as error:
            record = {"kind": kind, "passed": False, "error": f"{type(error).__name__}: {error}"}
        record["wall_seconds"] = round(time.monotonic() - start, 3)
        result["probes"].append(record)
    result["passed"] = result["advertised_tools_and_vision"] and all(p["passed"] for p in result["probes"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly permit the local model calls")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("Pass --run to execute local model probes")
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "data").resolve()):
        parser.error("Reports must remain in private data storage")
    if output.exists():
        parser.error("Choose a new report path; existing results are preserved")
    for model in args.models:
        runtime._validate_model_name(model)
    runtime.ensure_server()
    results = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        results.append(smoke_model(model))
        output.write_text(json.dumps({"models": results, "tools_executed": False}, indent=2), encoding="utf-8")
        print(json.dumps({"model": model, "passed": results[-1]["passed"],
                          "probes": [{"kind": p["kind"], "passed": p["passed"]} for p in results[-1]["probes"]]}), flush=True)
    return 0 if all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
