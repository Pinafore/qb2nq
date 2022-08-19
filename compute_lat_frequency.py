import numpy as np
import pandas as pd
import json
import os
import spacy
import argparse
# import neuralcoref
from collections import Counter, defaultdict

from question import Question

nlp = spacy.load('en_core_web_sm')
# neuralcoref.add_to_pipe(nlp)

class LatFrequencyComputer:
  def __init__(self, bad_tokens={"'s"}):
    self.page_frequency = defaultdict(Counter)
    self.bad_tokens = bad_tokens

  def count_answer_types(self, question, max_length=4):
    #-> Counter[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    lexical_answer_types = Counter()
    # Final all the 'this'
    for span in question.answer_nominal_mentions():
      span = list(span)
      if len(span) >= max_length:
        continue
      mention = " ".join(x.text for x in span[1:] if x.text not in self.bad_tokens)
      self.page_frequency[question.page][mention] += 1
      lexical_answer_types[mention] += 1

    if len(lexical_answer_types) > 0:
      return lexical_answer_types.most_common()[0][0]
    else:
      return ""

  def compute_lat_frequency(self, orig_qb_path: str, limit: int=-1) -> None:
    if os.path.exists(orig_qb_path) == False:
      print('Please check if {} exists in the current folder'.format(orig_qb_path))
    with open(orig_qb_path) as f1:
      qb_data = json.load(f1)['questions']

      for i in range(len(qb_data)):
        qid = qb_data[i]['qanta_id']
        page = qb_data[i]['page']
        text = qb_data[i]['text']
        lats = self.count_answer_types(Question(qid, page, text))
        # Printing here could cause unicode conversion error ifpage is not pure ASCII
        if i % 100 == 0:
          print("===> %i/%i: %s %s" % (i, len(qb_data), page, str(lats)))
        if limit > 0 and i > limit:
          break

  def most_common(self, page: str) -> str:
    """
    Return the most common lexical answer type for a page
    """
    if len(self.page_frequency[page]) == 0:
      return ""
    else:
      return self.page_frequency[page].most_common()[0][0]
  
  def write_most_freq_answer_type_for_qid(self, qanta_train_with_answer_type_path: str, output_file: str) -> None:
    if os.path.exists(qanta_train_with_answer_type_path) == False:
      print('Please check if {} exists in the current folder'.format(qanta_train_with_answer_type_path))
    qb_df = pd.read_json(qanta_train_with_answer_type_path, lines=True, orient='records')
      
    with open(qanta_train_with_answer_type_path) as f1:
      page_to_most_freq_answer_type_dict = {}
      qb_data = json.load(f1)['questions']

      for i in range(len(qb_data)):
        page_to_most_freq_answer_type_dict[qb_data[i]['qanta_id']] = self.most_common(qb_data[i]['page'])

    #save the most freq answer type for each qid into dictionary
    dir_name = os.path.dirname(output_file) 
    if not os.path.exists(dir_name): # create path if it doesn't exist
      os.makedirs(dir_name) 
    with open(output_file, 'w') as fp:
      json.dump(page_to_most_freq_answer_type_dict, fp, indent=2)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Apply heuristic functions")
  parser.add_argument('--limit', type=int,
                      default=-1,help="Limit of number of QB questions input")
  parser.add_argument('--qb_path', type=str,
                      default='qanta.train.2018.04.18.json',
                      help="path of the qb dataset")
  args = parser.parse_args()
  
  # load configuration 
  lat_freq_calculator = LatFrequencyComputer()
  lat_freq_calculator.compute_lat_frequency(args.qb_path, limit=args.limit)
  lat_freq_calculator.write_most_freq_answer_type_for_qid(args.qb_path, 'intermediate_results/lat_frequency.json')

