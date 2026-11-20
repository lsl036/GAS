import json
import tempfile
import unittest
from pathlib import Path

import run_quality_matrix as quality


class QualityMatrixTest(unittest.TestCase):
    def test_ruler_scores_use_the_correct_qa_rule(self):
        rows = [{"pred": "Paris", "outputs": ["Paris", "France"]}]
        self.assertEqual(quality.ruler_score("qa_1", rows), 100.0)
        self.assertEqual(quality.ruler_score("niah_single_1", rows), 50.0)
        self.assertEqual(quality.ruler_score("qa_1", [{"pred": "a\tb", "outputs": ["a\nb"]}]), 100.0)

    def test_resume_rejects_changed_source(self):
        source = [{"index": 1, "input": "question", "outputs": ["answer"]}]
        prediction = {"sample_index": 0, "source_sha256": quality.digest(source[0]), "pred": "answer",
                      "outputs": ["answer"]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.jsonl"
            path.write_text(json.dumps(prediction) + "\n")
            self.assertEqual(len(quality.validate_checkpoint(path, "ruler", "qa_1", source)), 1)
            with self.assertRaises(ValueError):
                quality.validate_checkpoint(path, "ruler", "qa_1", [{**source[0], "input": "changed"}])


if __name__ == "__main__":
    unittest.main()
