import unittest

from app.services.pdf_math_parser import (
    extract_requested_question_numbers,
    normalize_math_ocr_text,
    segment_hkdse_questions,
)


class TestPdfMathParser(unittest.TestCase):
    def test_normalize_math_ocr_text_symbols(self):
        raw = "∫ x dx = π/2 and a ≤ b, c ≠ d, θ"
        normalized = normalize_math_ocr_text(raw)
        self.assertIn("integral", normalized)
        self.assertIn("pi", normalized)
        self.assertIn("<=", normalized)
        self.assertIn("!=", normalized)
        self.assertIn("theta", normalized)

    def test_extract_question_numbers(self):
        self.assertEqual(extract_requested_question_numbers("Solve questions 1-3"), [1, 2, 3])
        self.assertEqual(extract_requested_question_numbers("For question 2 please"), [2])

    def test_segment_hkdse_questions_1_to_3(self):
        text = (
            "1. Find the value of x if 2x+3=11.\n"
            "2. In the diagram, triangle ABC is right-angled at B.\n"
            "3. Evaluate integral of x from 0 to 1.\n"
            "4. This should be ignored."
        )
        pages = [text]
        blocks = segment_hkdse_questions(text, pages, question_numbers=(1, 2, 3))
        self.assertEqual(set(blocks.keys()), {1, 2, 3})
        self.assertIn("Find the value of x", blocks[1].content)
        self.assertIn("triangle ABC", blocks[2].content)
        self.assertIn("Evaluate integral", blocks[3].content)


if __name__ == "__main__":
    unittest.main()
