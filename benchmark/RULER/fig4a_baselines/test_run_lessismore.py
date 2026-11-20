import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import run_lessismore as runner


class PairedResultsTest(unittest.TestCase):
    def test_score_and_duplicate_indices(self):
        self.assertEqual(runner.score({"pred": "FOO", "outputs": ["foo", "bar"]}), 50)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "formal" / runner.RELATIVE_PRED / "vt.jsonl"
            path.parent.mkdir(parents=True)
            rows = [{"index": 1, "input": text, "outputs": [text], "pred": text,
                     "kv_metrics": {"final_loaded_kv_tokens_mean": 100}}
                    for text in ("one", "two")]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            refs = runner.reference(root, "formal", "vt", 2)
            pred_path = root / "pred.jsonl"
            saved = [{"index": row["index"], "outputs": row["outputs"],
                      "input_sha256": hashlib.sha256(row["input"].encode()).hexdigest(),
                      "kv_metrics": row["kv_metrics"]} for row in refs]
            pred_path.write_text("".join(json.dumps(row) + "\n" for row in saved))
            self.assertTrue(runner.validate(pred_path, refs))
            with self.assertRaises(ValueError):
                runner.validate(pred_path, list(reversed(refs)))


if __name__ == "__main__":
    unittest.main()
