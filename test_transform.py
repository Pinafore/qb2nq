import unittest

from transform_question import QuestionRewriter

class TestQuestionTransformation(unittest.TestCase):
  def setUp(self):
    self.qr = QuestionRewriter(lat_frequency={},
                               min_length=3,
                               to_trim=["and", ","],
                               valid_verbs=["AUX"])

  def test_candidates(self):
      self.assertEqual(self.qr.generate_candidate_chunks_from_qb_question(
          "For 10 points, name this first American president"),
                       [(0, "For 10 points , name this first " +
                         "American president")])

      self.assertEqual(self.qr.generate_candidate_chunks_from_qb_question(
          "For 10 points, name this second American president and founder " +
          "of the University of Virginia"),
                       [(0, "For 10 points , name this second " +
                         "American president and founder of the " +
                         "University of Virginia"),
                        (0, "For 10 points , name this second " +
                         "American president"),
                        (0, "For 10 points , name ~ founder of the " +
                         "University of Virginia")])


  def test_trim(self):
    self.assertEqual(self.qr.trim_chunk("and he fasted"),
                     "he fasted")

    self.assertEqual(self.qr.trim_chunk(", and and he fasted ,"),
                     "he fasted")

    self.assertEqual(self.qr.trim_chunk("he fasted ,"),
                     "he fasted")


if __name__ == '__main__':
  unittest.main()
