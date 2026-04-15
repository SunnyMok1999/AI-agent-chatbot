import unittest

from app.services.marking_alignment import compute_marking_scheme_alignment


class TestMarkingAlignment(unittest.TestCase):
    def test_high_alignment_for_hkdse_style_answer(self):
        model_answer = (
            "Hint: isolate x.\n"
            "Steps:\n"
            "1) 2x + 3 = 11\n"
            "2) 2x = 8\n"
            "3) x = 4\n"
            "Final answer: x = 4."
        )
        scheme = [
            "2x + 3 = 11",
            "2x = 8",
            "x = 4",
        ]
        score = compute_marking_scheme_alignment(model_answer, scheme)
        self.assertGreaterEqual(score, 0.9)


if __name__ == "__main__":
    unittest.main()
