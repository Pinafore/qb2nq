import numpy as np
import pandas as pd
import json
import os
import spacy
# import neuralcoref
from collections import Counter, defaultdict

nlp = spacy.load('en_core_web_sm')
# neuralcoref.add_to_pipe(nlp)

class LatFrequencyComputer:
  def __init__(self):
    self.page_frequency = defaultdict(Counter)

  def count_answer_types(self, page: str, question: str, max_distance: int=4) -> Counter[str]:
    analysis = nlp(question)
    lexical_answer_types = Counter()
    # Final all the 'this'
    for this_idx in [idx for idx, token in enumerate(analysis) if token.text.lower() in ['this', 'these']]:
      # Find the first noun after "this"'s index
      try:
        first_noun = min(x for x in range(this_idx, min(this_idx + max_distance, len(analysis))) if analysis[x].pos_ == 'NOUN')
        next_non_noun = min(x for x in range(first_noun, min(first_noun + max_distance, len(analysis))) if analysis[x].pos_ != 'NOUN')
        lat = analysis[this_idx+1:next_non_noun].text.strip()
      except ValueError:
        lat = ""
      except IndexError:
        lat = ""

      self.page_frequency[page][lat] += 1
      lexical_answer_types[lat] += 1

    if len(lexical_answer_types) > 0:
      return lexical_answer_types.most_common()[0][0]
    else:
      return ""

  def compute_lat_frequency(self, orig_qb_path: str) -> None:
    if os.path.exists(orig_qb_path) == False:
      print('Please check if {} exists in the current folder'.format(orig_qb_path))
    with open(orig_qb_path) as f1:
      qb_data = json.load(f1)['questions']

      for i in range(len(qb_data)):
        page = qb_data[i]['page']
        text = qb_data[i]['text']
        lats = self.count_answer_types(page, text)
        if i%5000 == 0:
          print("===> %i/%i: %s %s" % (i, len(qb_data), page, str(lats)))

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

    if os.path.exists(orig_qb_path) == False:
      print('Please check if {} exists in the current folder'.format(orig_qb_path))
      
    with open(orig_qb_path) as f1:
      page_to_most_freq_answer_type_dict = {}
      qb_data = json.load(f1)['questions']

      for i in range(len(qb_data)):
        page_to_most_freq_answer_type_dict[qb_data[i]['qanta_id']] = self.most_common(qb_data[i]['page'])

    #save the most freq answer type for each qid into dictionary
    with open(output_file, 'w') as fp:
      json.dump(page_to_most_freq_answer_type_dict, fp, indent=2)

if __name__ == "__main__":
  orig_qb_path = 'qanta.train.2018.04.18.json'
  # load configuration 
  lat_freq_calculator = LatFrequencyComputer()
  lat_freq_calculator.compute_lat_frequency(orig_qb_path)
  lat_freq_calculator.write_most_freq_answer_type_for_qid(orig_qb_path, 'lat_frequency.json')

