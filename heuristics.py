import logging
import argparse
import json
from collections import Counter

import traceback
import re
from collections import defaultdict
from collections import Iterable

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk import Tree

import spacy

from nq_condition import *

nlp = spacy.load('en_core_web_sm')

# Heuristics for NQlike quality checking
class HeuristicsTransformer:
  def __init__(self, config, lat_lookup):
    self.current_analysis = {}    

    self.answer_type_dict = lat_lookup

    self.heuristics_ = [split_conjunctions("Split Connjunctions", config),
                          imperative_to_question("Imperative to Question", config),
                          remove_regexp_patterns("Remove Regexp", config),
                          drop_after_punctuation("Drop After Punc", config),
                          convert_continuous_to_present("Current Analysis", config),
                          no_wh_words("No WH words", config),
                          replace_this_is("Replace this is", config),
                          replace_which_with_this("Which with this", config),
                          which_none_is("Which none is", config),
                          what_is_which("What is which", config),
                          remove_rep_subject("Remove rep subject", config),
                          remove_BE_determiner("Remove be determiner", config),
                          add_space_before_punctuation("Space before punc", config),
                          rejoin_contractions("Rejoin conjunctions", config)]

  def cache_analysis(self, question, verbose=False):
    """
    Parse the sentence for downstream heuristics.  This prevents repeated computation in each of the heuristics.
    """
    tokens = nltk.word_tokenize(question.lower())
    text = nltk.Text(tokens)
    tagged = nltk.pos_tag(text)

    spacy_parse = nlp(question)
    if verbose:
      for ii in spacy_parse.sents:
        logging.debug(to_nltk_tree(ii.root))

    parses = {"spacy": spacy_parse, "nltk_tokens": tokens, "nltk_tags": tagged}
    self.current_analysis[question] = parses
    for ii in self.heuristics_:
      ii.add_analysis(question, parses)

    
  def __call__(self, qb_id, answer, chunk_id, question, lexical_answer_phrase, question_determiner, suppress_errors=False):
    self.current_analysis = {}
    self.cache_analysis(question)

    finished = set()
    applied_transformations = []

    while len(finished) < len(self.current_analysis):
      unchecked = [x for x in self.current_analysis if x not in finished]
      for qq in unchecked:
        assert not qq in finished
        for method in self.heuristics_:
          logging.debug("Applying method %s" % method.name)
          if suppress_errors:
            try:
              results = method(qb_id, qq, lexical_answer_phrase, question_determiner)
            except Exception as exc:
              logging.error(traceback.format_exc())
              logging.error(exc)
              results = []
              continue  
          else:
            results = method(qb_id, qq, lexical_answer_phrase, question_determiner)

          for new_question in results:
              if new_question != qq:
                logging.debug("Adding new question [%s]: %s" % (method_name, new_question))
                self.cache_analysis(new_question)

                row = {}
                row["qanta_id"] = qb_id
                row["original"] = question
                row["answer"] = answer
                row["parent"] = qq
                row["chunk_id"] = chunk_id
                row["question"] = new_question.lower()
                row["transform"] = method_name
                applied_transformations.append(row)
                
                self.cache_analysis(new_question)

          if method.replace:
            finished.add(qq)


    self.current_analysis = None
    return applied_transformations


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Demo heuristic functions")
  parser.add_argument('--qb_path', type=str,
                      default='qanta.train.2021.12.20.json',
                      help="path of the qb dataset")
  parser.add_argument('--config_file', type=str, default='config.json',
                      help="File with data that configures extraction")

  args = parser.parse_args()
  # Load dataset

  with open(args.config_file) as json_file:
    config = json.load(json_file)
  
  heuristics = HeuristicsTransformer(config, {0: "character", 1: "thing", 94: "ruler", 102: "novel", 104: "organ"})
  logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

  for qid, example in [(102, "An automobile in what novel was purchased in Fuller by Bessie for her marriage to 16-year - old Dude , and led to the death of both an African - American man and Grandmother Lester ."),
                       (94, "He demanded compensation for the family of Jacob Kaiser and forced another group to end its alliance with Austria in an armistice that he negotiated to end a war in which no battles occurred."),
                       (0, ' what character is first encountered in the Spouter - Inn where the landlord thinks he may be late because " he ca n\'t sell his head , " and his coffin helps save the narrator after the ship he \'s on sinks . \xa0'),
                       (19, "The protagonist of one of who 's works gives a jar to his friend instead of repaying a loan , and later dies after embezzling money in an attempt to buy out a courtesan 's contract ."),
                           (1, "'!$  % ##@# @@# ftp FTP ftp--- For 10 points, for 10 points , For 10 points --- for 20 points: for 10 points--this is a real, legit question !@#!@@#@%%%....?"),
                           (104, "For 10 points , name this organ , home to the islets of Langerhans ."),
                           (8, "For 10 points , identify this French government that succeeded the Second Empire ."),
                           (98, "For 10 points , identify this Sanskrit - language author of \" The Cloud Messenger \" and The Recognition of Shakuntala ."),
                           (59, "For 10 points , name this mortal lover of Zeus and mother of Dionysus ."),
                           (20, "For 10 points , name this Italian author of The Name of the Rose and Foucault 's Pendulum ."),
                           (17, "For 10 points , name this country , the home of the director of The Seventh Seal , Ingmar Bergman ."),
                           (119, "For 10 points , name this 1859 essay condemning infringements on personal freedom , written by John Stuart Mill ."),
                           (378, "Beginning with \" April is the cruellest month \" , for 10 points , name this long poem in five sections by T.S. Eliot ."),
                           (361,  " who asserted \" the air is a spongy body / a promiscuous faceless being \" in \" The Balcony \" and asked \" Do I believe in man / or in the stars ? \" in another poem .")]:

      transformations = heuristics(qid, "??", 0, example, "what thing", "what")
      for transform in transformations:
        logging.debug("===============================")
        logging.debug(transform["original"])
        logging.debug(transform["parent"])
        logging.debug(transform["transform"])           
        logging.debug(transform["question"])
   

  
