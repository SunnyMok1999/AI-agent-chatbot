import unittest

from app.services.pdf_math_parser import segment_hkdse_questions


class TestQuestionBoundaryDetection(unittest.TestCase):
    def test_detect_boundaries_with_ocr_noise(self):
        noisy_text = (
            "HKDSE 2012 Core Mathematics\n"
            "1) If \\frac{1}{2}x = 3, find x.\n"
            "Diagram omitted\n"
            "2) In Figure 1, AB ⟂ BC and AB = 4.\n"
            "Use Pythagoras to find AC.\n"
            "3) Evaluate ∑_{k=1}^{3} k.\n"
        )
        blocks = segment_hkdse_questions(noisy_text, [noisy_text], question_numbers=(1, 2, 3))
        self.assertEqual(len(blocks), 3)
        self.assertTrue(blocks[1].content.startswith("1)"))
        self.assertIn("AB", blocks[2].content)
        self.assertIn("Evaluate", blocks[3].content)


if __name__ == "__main__":
    unittest.main()
