"""Functional checks for trusted synthetic training preparation, no ML dependencies."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import prepare_training as prep


class TrainingPreparationTests(unittest.TestCase):
    def test_all_fixtures_fail_before_and_pass_after_with_focused_edit(self):
        for family in prep.FAMILIES:
            for variant in range(6):
                with self.subTest(family=family, variant=variant):
                    row, verification = prep.verified_example(prep.fixture(family, variant))
                    self.assertEqual(verification["before_exit_code"], 1)
                    self.assertEqual(verification["after_exit_code"], 0)
                    names = {tool["function"]["name"] for tool in row["tools"]}
                    waiting = []
                    for message in row["messages"]:
                        if message["role"] == "tool":
                            self.assertEqual(message["name"], waiting.pop(0))
                            json.loads(message["content"])
                        else:
                            self.assertFalse(waiting)
                            for call in message.get("tool_calls", []):
                                self.assertIn(call["function"]["name"], names)
                                self.assertIsInstance(call["function"]["arguments"], dict)
                                waiting.append(call["function"]["name"])
                    self.assertFalse(waiting)

    def test_reference_that_does_not_fix_the_bug_is_rejected(self):
        case = prep.fixture("run_length", 0)
        case["after"] = case["before"]
        with self.assertRaises(ValueError):
            prep.verified_example(case)

    def test_generated_bundle_is_deterministic_and_explicitly_untrained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = prep.prepare(root)
            original = (root / "training_bundle.zip").read_bytes()
            prep.prepare(root)
            self.assertEqual(original, (root / "training_bundle.zip").read_bytes())
            self.assertEqual(result["status"], "NOT_TRAINED")
            self.assertFalse(result["heldout_benchmark_included"])
            self.assertEqual((result["training_examples"], result["validation_examples"]), (16, 8))
            with zipfile.ZipFile(root / "training_bundle.zip") as bundle:
                manifest = json.loads(bundle.read("manifest.json"))
                for name, digest in manifest["files"].items():
                    self.assertEqual(hashlib.sha256(bundle.read(name)).hexdigest(), digest)
                self.assertEqual(set(bundle.namelist()), {"train.jsonl", "validation.jsonl", "manifest.json"})
            notebook = json.loads((root / "local-agent-4b-training-pilot.ipynb").read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    code = "".join(line for line in cell["source"] if not line.lstrip().startswith("%pip"))
                    ast.parse(code)
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
            code = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
            self.assertIn("load_in_4bit=False", code)
            self.assertIn("EXPORT_CANDIDATE = False", code)
            self.assertIn(prep.MODEL_REVISION, code)
            self.assertIn("processing_class=text_tokenizer", code)
            self.assertNotIn("tokenizer=tokenizer", code)
            self.assertIn("from unsloth import FastVisionModel", code)
            self.assertIn("text_only=False", code)
            self.assertIn("finetune_vision_layers=False", code)


if __name__ == "__main__":
    unittest.main()
