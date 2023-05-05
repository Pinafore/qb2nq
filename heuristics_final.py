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
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet as wn
from nltk import Tree

import spacy

from nq_condition2 import *

nlp = spacy.load('en_core_web_sm')

# Heuristics for NQlike quality checking
class HeuristicsTransformer:
  def __init__(self, config, lat_lookup):
    self.current_analysis = {}    

    self.answer_type_dict = lat_lookup

    self.heuristics_ = [split_conjunctions("Split Connjunctions", config),
                          imperative_to_question("Imperative to Question", config),
                          no_wh_words("No WH words", config)]

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
                logging.debug("Adding new question [%s]: %s" % (method.name, new_question))
                self.cache_analysis(new_question)

                row = {}
                row["qanta_id"] = qb_id
                row["original"] = question
                row["answer"] = answer
                row["parent"] = qq
                row["chunk_id"] = chunk_id
                row["question"] = new_question.lower()
                row["transform"] = method.name
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
  parser.add_argument('--output_file', type=str, default='./qanta_heuristic_output.json',
                      help="File with data that configures extraction")

  args = parser.parse_args()
  # Load dataset

  with open(args.config_file) as json_file:
    config = json.load(json_file)

  with open(args.qb_path) as json_file:
    questions_full = json.load(json_file)

  example=[]
  qid=[]
  exs=[]
  for i in questions_full:
    question=i['text']
    qid_i=i['qanta_id']
    answer=i["answer"]
    question_split=sent_tokenize(question)
    for j in range(len(question_split)):
      e1=tuple((qid_i,answer,question_split[j]))
      exs.append(e1)
      #qid.append(qid_i)
      #example.append(question_split[i])
    #print(question)
  
  heuristics = HeuristicsTransformer(config, {0: "character", 1: "thing", 94: "ruler", 102: "novel", 104: "organ"})
  logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

  f1 = open(args.output_file, "w")
  f1.write("[")
  for qid, answer,example in exs:
      doc=nlp(example)
      delt=" "
      for i in doc.noun_chunks:
        if i.text.lower().find("this ")!=-1:
          delt=i.text.lower().replace("this ","which ")
          break
        elif i.text.lower().find("these ")!=-1:
          delt=i.text.lower().replace("these ","which ")
          break
        for j in doc:
          if j.text.lower()=='he':
            delt=j.text.lower().replace("he","who")
            break
          elif j.text.lower()=='she':
            delt=j.text.lower().replace("she","who")
            break
          elif j.text.lower()=='they':
            delt=j.text.lower().replace("they","what")
            break
        break
      print("replaced text: ",delt)
      transformations = heuristics(qid, answer, 0, example, "what thing", delt)
      json.dump(transformations,f1,indent=4)
      f1.write(",")
      for transform in transformations:
        logging.debug("===============================")
        logging.debug(transform["original"])
        logging.debug(transform["parent"])
        logging.debug(transform["transform"])           
        logging.debug(transform["question"])
        print(transform["question"])
  f1.write("]")
  f1.close()
