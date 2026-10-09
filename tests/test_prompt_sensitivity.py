"""Unit checks for synthetic Prompt-change diagnostics."""
import unittest

from tools.diagnose_prompt_sensitivity import make_negative, summarize


class PromptSensitivityTests(unittest.TestCase):
    def test_subject_and_action_are_changed_independently(self):
        prompt = "A Cat is eating dinner"
        self.assertEqual(make_negative(prompt, "subject"), ("A Dog is eating dinner", "cat->dog"))
        self.assertEqual(make_negative(prompt, "action"), ("A Cat is cooking dinner", "eating->cooking"))

    def test_ambiguous_or_already_present_targets_are_rejected(self):
        self.assertIsNone(make_negative("a cat and a dog", "subject"))
        self.assertIsNone(make_negative("a cat and a car", "subject"))
        self.assertIsNone(make_negative("a cat is eating while sitting", "action"))
        self.assertIsNone(make_negative("a landscape", "subject"))

    def test_summary_keeps_coverage_and_gap_direction_separate(self):
        rows = [
            {"checkpoint": "old", "kind": "subject", "gap": 0.2},
            {"checkpoint": "old", "kind": "subject", "gap": -0.1},
            {"checkpoint": "new", "kind": "subject", "gap": 0.3},
        ]
        summary = summarize(rows, eligible_count=4)
        old = next(row for row in summary if row["checkpoint"] == "old" and row["kind"] == "subject")
        self.assertEqual(old["candidate_pairs"], 2)
        self.assertEqual(old["coverage"], 0.5)
        self.assertEqual(old["positive_gap_rate"], 0.5)
        self.assertAlmostEqual(old["mean_gap"], 0.05)


if __name__ == "__main__":
    unittest.main()
