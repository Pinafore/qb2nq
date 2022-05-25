#Step1. import libraries

import sys

import logging
import numpy as np
import pandas as pd
import json
import string

import time
import os
import re
import random
import argparse

import logging
from collections import defaultdict

from collections import Counter

from typing import List, Tuple

from question import Question
from heuristics import HeuristicsTransformer
from answer_type import AnswerType

# Main

class QuestionRewriter:

  def __init__(self, lat_frequency, min_length, to_trim, valid_verbs, remove_dict, non_last_sent_transform_dict, heuristics, answer_classifier):
    self.lat_frequency = lat_frequency

    # Minimum number of tokens in a chunk
    self.min_length = min_length

    self.valid_verbs = valid_verbs
    self.remove_dict = remove_dict
    self.non_last_sent_transform_dict = non_last_sent_transform_dict
    self.heuristics = heuristics
    
    self.answer_classifier = answer_classifier

  def single_question_transform(self, question):
    # parse tree
    qb_id = question.qid
    nq_like_questions = []
    orig_output_before_transformation = []

    lexical_answer_type = self.lat_frequency[qb_id]
    pronoun = self.answer_classifier.question_word(lexical_answer_type)
    question_determiner = self.answer_classifier.determiner(lexical_answer_type)
    lexical_answer_phrase = "%s %s" % (question_determiner, lexical_answer_type)
    
    # generate candidates from qb_question
    chunks = [" ".join(x) for x in question.generate_chunks(lexical_answer_phrase, pronoun)]
    chunks += [" ".join(x) for x in question.sentences()]
    logging.debug("==============================")
    logging.debug("INITIAL %i CHUNKS| LAT: %s PRO: %s DET: %s" % (len(chunks), lexical_answer_type, pronoun, question_determiner))    

    
    tranformation_rows = []

    for chunk, original in enumerate(chunks):
        transformations = self.heuristics(qb_id, chunk, original, lexical_answer_phrase, question_determiner)

        for candidate in transformations:
            tranformation_rows.append(candidate)
    return tranformation_rows

  def transform_questions(self, input_file, limit):
    with open(input_file) as infile:
      questions = json.load(infile)
      if limit > 0:
        questions = questions['questions'][:limit]
      else:
        questions = questions['questions']

    transformed = []
    for row in questions:
      question = Question(row['qanta_id'], row['page'], row['text'])
      transformed += self.single_question_transform(question)

    return transformed
    
  # helper functions
  def capitalization(self, q):
    q = q[0].upper()+q[1:]
    return

  def remove_duplicates(self, q):
    words = q.split()
    for i, w in enumerate(words):
      if i >= (len(words)-1):
        continue
      w2 = words[i+1]
      w2 = re.sub('\'s', '', w2)
      if w == w2:
        words = words[:i]+words[i+1:]
    q = " ".join(words)
    return q


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Apply heuristic functions")
  parser.add_argument('--limit', type=int,
                      default=-1,help="Limit of number of QB questions input")
  parser.add_argument('--qb_path', type=str,
                      default='qanta.train.2021.12.20.json',
                      help="path of the qb dataset")
  parser.add_argument('--lat_freq', type=str, default='intermediate_results/lat_frequency.json',
                      help="JSON of frequency for each LAT")
  parser.add_argument('--raw_text_output', type=bool, default=True,
                      help="Save both the raw output before transformation and the transformed questions")
  parser.add_argument('--min_chunk_length', type=int, default=5,
                      help="How long must extracted segment of QB question be?")
  parser.add_argument('--config_file', type=str, default='config.json',
                      help="File with data that configures extraction")
  parser.add_argument('--log_level', type=str, default='')
  parser.add_argument('--nqlike_output', type=str, default='intermediate_results/nq_like.json',
                      help="Where we write transformed questions")

  args = parser.parse_args()
  # Load dataset

  if args.log_level=='debug':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
  
  qb_path = args.qb_path
  limit = args.limit
  qb_df = None

  # read LAT frequency
  with open(args.lat_freq) as json_file:
    lat_frequency = json.load(json_file)

  for ii in list(lat_frequency.keys()):
    lat_frequency[int(ii)] = lat_frequency[ii]
  logging.info("Loaded %i LAT keys, e.g.: %s" % (len(lat_frequency), str(lat_frequency.keys())[:120]))

  # read contents from config.json file
  with open(args.config_file) as json_file:
    config = json.load(json_file)

  transformer = HeuristicsTransformer(config, lat_frequency)

  rewriter = QuestionRewriter(lat_frequency,
                              min_length=args.min_chunk_length,
                              to_trim=config["to_trim"],
                              valid_verbs=config["valid_verbs"],
                              remove_dict=config["remove_dict"],
                              non_last_sent_transform_dict=config["non_last_sent_transform_dict"],
                              heuristics=transformer,
                              answer_classifier=AnswerType())
  
  transformed = rewriter.transform_questions(qb_path, limit)
  with open(args.nqlike_output, 'w') as outfile:
      json.dump(transformed, outfile, indent=2)

  # # prepare NQlike and QB with contexts datasets for the classifier retraining QA
  # qb_id_list = qb_id_input.tolist()
  # qb_df = pd.read_json(qb_path, lines=True, orient='records')
  # selected_qb_df = qb_df.loc[qb_df.apply(lambda x: x.qanta_id in qb_id_list, axis=1)]
  # selected_qb_df = selected_qb_df.rename(columns={'score': 'quality_score'})
  # # save QB_with_contexts
  # selected_qb_df.to_json('./qb_with_contexts.json', orient='index', indent=2)
  # # mapping nq_like with contexts
  # context_list = []
  # char_spans_list = []
  # answer_list = []
  # for idx in qb_id_list:
  #       context_list.append(selected_qb_df.loc[selected_qb_df['qanta_id'] == idx]['context'])
  #       char_spans_list.append(selected_qb_df.loc[selected_qb_df['qanta_id'] == idx]['char_spans'])
  #       answer_list.append(selected_qb_df.loc[selected_qb_df['qanta_id'] == idx]['answer'])
