import unittest
from src.automatic_prompt_generator import generate_answer

class TestRankedPromptsGenerator(unittest.TestCase):
  def test_generate_answer(self):
    prompt = "What is RAG?"
    answer = generate_answer(prompt)

    # Add assertions here to check the answer
    self.assertIsNotNone(answer, "Answer should not be None")
    self.assertIsInstance(answer, str, "Answer should be a string")
    # Add more assertions based on your specific expectations

if __name__ == '__main__':
  unittest.main()