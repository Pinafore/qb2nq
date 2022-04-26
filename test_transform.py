import unittest
import json
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
from transform_question import QuestionRewriter, HeuristicsTransformer, AnswerTypeClassifier

class TestQuestionTransformation(unittest.TestCase):
  def setUp(self):
    with open('config.json') as json_file:
      config = json.load(json_file)
    tokenizer = DistilBertTokenizerFast.from_pretrained(config["tokenizer"])
    loaded_model = TFDistilBertForSequenceClassification.from_pretrained(config["loaded_model"])
    with open(config["last_sent_word_transform_30000"], 'r') as f:
      last_sent_word_transform_30000 = json.load(f)
    self.transformer = HeuristicsTransformer(config)
    self.answerTypeClassifier = AnswerTypeClassifier(last_sent_word_transform_30000=last_sent_word_transform_30000, tokenizer=tokenizer, loaded_model=loaded_model)
    
    self.qr = QuestionRewriter(lat_frequency={},
                               min_length=3,
                               to_trim=["and", ","],
                               valid_verbs=["AUX"],
                               remove_dict=config["remove_dict"],
                               non_last_sent_transform_dict=config["non_last_sent_transform_dict"],
                               heuristicsTransformer=self.transformer,
                               answerTypeClassifier=self.answerTypeClassifier)
    

  def test_candidates(self):
      self.assertEqual(self.qr.single_question_transform(1, 
          'The oldest document written in this language is a letter written in 1521 in the town of Câmpulung, while more recent poets writing in this language include Carmen Sylva and Anton Pann. This language uses five cases, though the genitive and dative cases are identical, as are the nominative and accusative. Tripthongs occur frequently in this language, as in "rusaoică," while interjections in this language include "mamă-mamă." It is more closely related to Dalmatian than to Italian or Spanish, and this language includes the pronouns "noi," "voi," and "eu" ["AY-oo"] and favors labial consonants such as "b" and "m" over velars such as "g" and "k." For 10 points, name this tongue spoken by the members of O-Zone and Nicolae Ceauşescu, an Eastern Romance language spoken in Bucharest.')[:3],
                       [{'nq_like_questions': 'the oldest document written in which language is a letter written in 1521 in the town of câmpulung', 'orig_output_before_transformation': 'The oldest document written in this language is a letter written in 1521 in the town of Câmpulung , .'}, {'nq_like_questions': 'while more recent poets writ in which language include carmen sylva and anton pann', 'orig_output_before_transformation': 'while more recent poets writing in this language include Carmen Sylva and Anton Pann'}, {'nq_like_questions': 'which language uses five cases', 'orig_output_before_transformation': 'This language uses five cases , .'}])

  def test_trim(self):
    self.assertEqual(self.qr.trim_chunk("and he fasted"),
                     "he fasted")

    self.assertEqual(self.qr.trim_chunk(", and and he fasted ,"),
                     "he fasted")

    self.assertEqual(self.qr.trim_chunk("he fasted ,"),
                     "he fasted")
  
  def test_heuristic(self):
    self.assertEqual(self.transformer.clean_marker(" )which german philosopher is this philosopher wrote a work , . "),
                     "which german philosopher is this philosopher wrote a work")

    self.assertEqual(self.transformer.clean_answer_type("-- name this person who presented a proposal"),
                     "which person who presented a proposal")

    self.assertEqual(self.transformer.drop_after_semicolon("Henry's words were not transcribed; but no one forgot their eloquence"),
                     "Henry's words were not transcribed")

    self.assertEqual(self.transformer.remove_pattern("for 10 pointsNo man thinks moreÃƒ highly than I do"),
                     "No man thinks more highly than I do")

    self.assertEqual(self.transformer.remove_rep_subject("Why man is this man often see the same subject in different lights"),
                     "Why man often see the same subject in different lights")

    self.assertEqual(self.transformer.remove_BE_determiner("I shall speak forth man is his sentiments"),
                     "I shall speak forth man's sentiments")

    self.assertEqual(self.transformer.fix_no_verb("which north god wielding north god"),
                     "which north god is wielding north god")

    self.assertEqual(self.transformer.remove_name_which("identify which anguish of spirit it may cost"),
                     "which anguish of spirit it may cost")
    
    self.assertEqual(self.transformer.no_wh_words(-1, "this play begins with the protagonist"),
                     "which play begins with the protagonist")
                     
    self.assertEqual(self.transformer.this_is_pattern(96407, "this is known as uniform motion."),
                     "which law is known as uniform motion.")

    self.assertEqual(self.transformer.remove_end_be_verbs("which jewish holiday is that hymn is"),
                     "which jewish holiday is that hymn")

    self.assertEqual(self.transformer.remove_extra_AUX("which composer is are we a pair"),
                     "which composer are we a pair")   

    self.assertEqual(self.transformer.WDT_BE_pattern("michael green is a current professor at this university , which is where watson and crick discovered dna 's structure"),
                     "michael green is a current professor at which university , that is where watson and crick discovered dna 's structure")
    
    self.assertEqual(self.transformer.no_WDT_and_WRB(115148,"its modern appellation"),
                     "which English explorer is its modern appellation")  

    self.assertEqual(self.transformer.VERB_AUX_at_beginning(152819,"were refused real employment because of logical discrimination"),
                     "which se people were refused real employment because of logical discrimination")  

    self.assertEqual(self.transformer.which_none_is(79908,"which none is its longest sections"),
                     "which holy text is its longest sections")  
    
    self.assertEqual(self.transformer.add_space_before_punctuation("welcome to, the jungle\'s"),
                     "welcome to , the jungle \'s")  


if __name__ == '__main__':
  unittest.main()
