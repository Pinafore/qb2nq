import logging

import nltk
import numpy as np
import pandas as pd
import re
from collections import defaultdict
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# import textstat
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from math import log
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
import sklearn.model_selection
import sklearn.preprocessing as preproc
from sklearn.feature_extraction import text
from sklearn import tree
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
from collections import Counter
import pickle
import warnings
warnings.filterwarnings("ignore")
import nltk
import argparse
from scipy.sparse import hstack


# word2vec library
import gensim.downloader
import gensim.models
from gensim.test.utils import common_texts
# Word2Vec initialization
# w2v_model = gensim.downloader.load('glove-wiki-gigaword-300')

# unigram initialization

#unigram_converter = CountVectorizer(max_features=MAXFEATURES)




# Step 6: Functions to extract features

def read_nq_like(location):
  print("Reading NQ questions from %s" % location)
  with open(location) as infile:
    records = json.loads(infile.read())
    questions = pd.DataFrame.from_records(records)
  questions['label'] = 1
  questions['source'] = location
  return questions

def build_dataset(nq_location: str, nqlike_location:str, limit:int=-1, percent_test:float=0.25):
  nq_like = read_nq_like(nqlike_location)
  nq = pd.read_json(nq_location, lines=True, orient='records')
  nq['label'] = 0
  nq['source'] = nq_location

  print("=====NQ=====")
  print(nq.head())
  print("=====NQ Like=====")
  print(nq_like.head())

  if limit > 0:
    fold_size = min(limit, nq.shape[0], nq_like.shape[0])
  else:
    fold_size = min(nq.shape[0], nq_like.shape[0])
  print("Balanching to have %i questions" % fold_size)
    
  merged = pd.concat([nq_like.sample(fold_size), nq.sample(fold_size)], axis=0, ignore_index=True)
  merged['bigram'] = merged['question'].apply(lambda x: "START %s STOP" % x)
  nq_like['bigram'] = nq_like['question'].apply(lambda x: "START %s STOP" % x)
  
  train, test = sklearn.model_selection.train_test_split(merged, train_size=1.0-percent_test, random_state=42)
  
  nq_len = get_NQ_length_percentile(nq)

  return train, test, nq_like, nq_len

def binarize(x):
    if x < 1:
        return 0
    else:
        return 1
    
# function to get number of nouns
def count_num_nouns(text):
    tokens = nltk.word_tokenize(text.lower())
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)
    counts = Counter(tag for word,tag in tags)
    noun_num = counts['NN']
    return noun_num

def count_unique_words(x):
      counter = Counter(x.lower().split())
      return sum(1 for x in counter if counter[x] >= 1)
    
def count_max_duplicates(x):
      counter = Counter(x.lower().split())
      if len(counter.values()) > 0:
        return max(counter.values())
      else:
        return 0

# function to get number of verbs
def count_num_verbs(text):
    tokens = nltk.word_tokenize(text.lower())
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)
    counts = Counter(tag for word,tag in tags)
    verb_num = counts['VB']
    return verb_num

# function compute maximum idf score feature
def extract_terms( document ):
   terms = document.lower().replace('?',' ').replace('.',' ').replace(',',' ').split()
   return terms

def calculate_idf( documents ):
   N = len(documents)
   from collections import Counter
   tD = Counter()
   for d in documents:
      for f in d:
          tD[f] += 1
   IDF = {}
   import math
   for (term,term_frequency) in tD.items():
       term_IDF = math.log(float(N) / term_frequency)
       IDF[term] = term_IDF
   return IDF
 
def num_of_words(qq):
  return len(qq.split())

def get_NQ_length_percentile(df):
  num_list = df['question'].apply(num_of_words).values
  return (num_list[round(len(num_list)*0.05)], num_list[round(len(num_list)*0.95)])

class Word2Vec:
  def __init__(self, model=gensim.downloader.load('glove-wiki-gigaword-300')):
    self.w2v_model = model
  # Step 12
  # function average word2vec vector
  def avg_feature_vector(self,words, model, num_features, ind2key_set):
      feature_vec = np.zeros((num_features, ), dtype='float32')
      n_words = 0
      for word in words:
          if word in ind2key_set:
              n_words += 1
              feature_vec = np.add(feature_vec, model[word])
      if (n_words > 0):
          feature_vec = np.divide(feature_vec, n_words)
      return feature_vec
      
  # define vectorizer
  def word2vec_vectorizer(self,data, model,num_features,ind2key_set):
      sentence = data.lower().split(' ')
      return self.avg_feature_vector(sentence,model,num_features,ind2key_set)

  # vectorize function
  def vectorize(self,question):
      return self.word2vec_vectorizer(question,self.w2v_model,300,set(self.w2v_model.index_to_key))

  # reshape word2vec vectors
  def reshape_all(self,w2v_vectors):
      a = len(w2v_vectors)
      b = 300
      all_vectors = []
      for v in w2v_vectors:
          for e in v:
              all_vectors.append(e)
      all_vectors = np.reshape(all_vectors, (a, b))
      return all_vectors
  
    
class Classifier:
  def __init__(self, feature_list, abnormal_length=4, w2v=Word2Vec(), max_term_features = 20, nq_len = None):
    self._length_cutoff = abnormal_length
    self._features = feature_list
    self.feature_names = []
    self.w2v = w2v
    self.tokenizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), max_features=max_term_features, lowercase=False)
    self.nq_len = nq_len

  def initialize_tokenizer(self, train_set):
    self.tokenizer.fit(train_set['bigram'])
    
  def get_bigrams(self, df_train):
    return self.tokenizer.transform(df_train['bigram'])
    
  def get_ablength(self, df_train):
    values = df_train['question'].apply(lambda x: len(x.split()) < self._length_cutoff).values
    values = values.reshape([len(values), 1])
    return values
    
  # function to compute length feature
  def get_length(self, df_train):
    values = df_train['question'].apply(lambda x: log(1 + len(x.split()))).values
    num_rows = len(values)
    values = values.reshape([num_rows, 1])
    return coo_matrix(values, shape=[num_rows, 1])

  #function to compute kincaid readability score
  def get_kincaid(self, df_train):
    return df_train['question'].apply(textstat.flesch_kincaid_grade).values

  # function to compute duplicate feature
  def get_uniques(self, df_train):    
    values = df_train['question'].apply(count_unique_words).values
    values = values.reshape([len(values), 1])
    return values
  
  def get_num_nouns(self, df_train):
    return df_train['question'].apply(count_num_nouns).values

  def get_num_verbs(self, df_train):
    return df_train['question'].apply(count_num_verbs).values

  def get_max_idf(self, df_train):
    documents = df_train['question'].apply(extract_terms)
    IDF = calculate_idf(documents)
    max_idf = []
    for doc in documents:
      idf_lst = []
      for t in doc:
        idf_lst.append(IDF[t])
      if len(idf_lst) != 0:
        max_idf.append(max(idf_lst))
      else:
        max_idf.append(len(documents))
    return max_idf
  
  def get_max_duplicates(self, df_train):
    values =  df_train['question'].apply(count_max_duplicates).values
    values = values.reshape([len(values), 1])
    return values
  
  def is_in_nq_length_percentile_5(self, x):
    lower, upper = self.nq_len
    num = num_of_words(x)
    if num >= lower:
      return 1
    else:
      return 0
    
  def is_in_nq_length_percentile_95(self, x):
    lower, upper = self.nq_len
    num = num_of_words(x)
    if num <= upper:
      return 1
    else:
      return 0
    
  def get_percentile_length_5(self, df_train):
    values = df_train['question'].apply(self.is_in_nq_length_percentile_5).values
    values = values.reshape([len(values), 1])
    return values
  
  def get_percentile_length_95(self, df_train):
    values = df_train['question'].apply(self.is_in_nq_length_percentile_95).values
    values = values.reshape([len(values), 1])
    return values

  def get_word2vec(self, df_train):
    return df_train['question'].apply(self.w2v.vectorize).values

  def feature_name(self, feature):
    if feature == "bigrams":
      lookup = self.tokenizer.vocabulary_
      lookup = sorted((y, x) for x, y in lookup.items())
      lookup = [y for x, y in lookup]
      
      return lookup
    else:
      return [feature]

  def prepare_features(self, df):
    self.feature_names = []
    features = []

    for ii in self._features:
      logging.info("Building feature %s" % ii)
      method = getattr(self, "get_" + ii)
      names = self.feature_name(ii)
      for ii in names:
        self.feature_names.append(ii)
      features.append(method(df))

    return hstack(features)

  # helper function to prepare data for classifier
  def get_x_vals(self, dictionary):
    x_vals = np.zeros((len(dictionary['label']), 1))
    for key in dictionary.keys():
      if key != 'label' and key != 'qanta_id' and key != 'word2vec':
        if key == 'unigram':
          x = [' '.join(it) for it in dictionary['unigram']]
          x = self.unigram_converter.transform(x)
          x_vals = np.concatenate((x_vals, x.toarray()), axis=1)
        else:
          x = np.reshape(dictionary[key], (-1, 1))
          x_vals = np.concatenate((x_vals, x), axis=1)
  
    if 'word2vec' in dictionary.keys():
        new_x_vals = []
        all_vectors = self.w2v.reshape_all(dictionary['word2vec'])
        for i in range(0,len(all_vectors)):
          new_x_vals.append(np.concatenate((x_vals[i],all_vectors[i])))
        x_vals = new_x_vals
    x_vals = np.delete(x_vals, 0, 1)
    return x_vals

  # function to train classifier    
  def train_classifier(self, train):
    y_train = train['label'].values
    x_train = self.prepare_features(train)
    model = LogisticRegression().fit(x_train,y_train)
    return model

  # function to evaluate classifier
  def evaluate(self, model, df_test):
    x = self.prepare_features(df_test)
    results = model.predict_proba(x)
    df_test['prob_scores'] = results[:, 1]
    predictions = model.predict(x)
    df_test['pred'] = predictions
    
    # TODO: Compute precision and recall
    #
    # of the things that we say are NQ, how many actually are?
    # of the NQ questions, how many did we find?

    acc = accuracy_score(predictions, df_test['label'])
    logging.info('Accuracy: '+format(acc)+'\n')
    return df_test

  def generate_feature_weight(self, model):
    feature_weight = {}

    assert len(self.feature_names) == len(model.coef_[0]), "Names do not match feature dimension (%i vs %i)" % (len(self.feature_names), len(model.coef_[0]))
    for name, weight in zip(self.feature_names, model.coef_[0]):
      feature_weight[name] = weight

    feature_weight["BIAS"] = model.intercept_[0]
    return feature_weight

  def save_dictionary(self, questions, file_path):   
    x = self.prepare_features(questions)
    results = model.predict_proba(x)

    questions['score'] = results[:, 1]

    dict_to_write = defaultdict(list)

    question_ids = set(questions['qanta_id'])
    for qq in question_ids:
      chunks = set(nq_like[nq_like.qanta_id == qq].chunk_id)
      for chunk in chunks:
        # Find the highest scoring prediction, take that as answer
        subview = questions[(questions.qanta_id == qq) & (questions.chunk_id == chunk)]
        best_score = min(subview.score)

        for row_id, row in subview[subview.score == best_score].iterrows():
          dict_to_write['qanta_id'].append(row.qanta_id)
          dict_to_write['score'].append(row.score)          
          dict_to_write['question'].append(row.question)
          dict_to_write['original'].append(row.original)
          dict_to_write['chunk_id'].append(row.chunk_id)
          break
    
    df = pd.DataFrame(data=dict_to_write)
    df.to_csv(file_path)


  
# Step 9

# flag = 0 (0 flag refers to wellformedness classifier with an accuracy score) 
# flag = 1 (1 flag refers to NQ-like classifier with 0 accuracy score as we do not have labeled qualityscore for NQ-like)
# qb_last = True: only consider the last sentence of the qb questions
# qb_last = False: consider the whole qb questions
if __name__=="__main__":
  logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
  parser = argparse.ArgumentParser(description="Create classifier to discriminate synthetic questions from real questions")
  parser.add_argument('--limit', type=int, default=-1)
  parser.add_argument('-f', '--feature_list', nargs='+', default=['uniques','max_duplicates', 'percentile_length_5','percentile_length_95', 'bigrams'])
  parser.add_argument('--predictions', type=str, default='intermediate_results/nqlike_scores.csv')
  parser.add_argument('--nq_data', type=str, default='TriviaQuestion2NQ_Transform_Dataset/NaturalQuestions_train_reformatted.json')
  parser.add_argument('--nqlike_data', type=str, default='intermediate_results/nqlike_train.json')  
  parser.add_argument('--max_term_features', type=int, default=50)
  parser.add_argument('--seq', type=str, default='')
  args = parser.parse_args()
	# set flag and if_qb_last_sent here
	# 0 --wellformedness accuracy output
	# 1 --NQ-like output

  nqlike_path = 'nqlike_train/{}.json'.format(args.seq)
  train, test, nq_like, nq_len = build_dataset(args.nq_data, nqlike_path, args.limit)

  c = Classifier(args.feature_list, max_term_features=args.max_term_features, nq_len=nq_len)
  c.initialize_tokenizer(train)
  
  # TODO: Would be good to do cross-fold validation for the NQ like predictions

  # Train model and output the weights
  model = c.train_classifier(train)
  weight_dict = c.generate_feature_weight(model)
  
  c.evaluate(model, test)
  logging.info("Weight dict", weight_dict)
  with open('intermediate_results/logistic_regression_weight_dict_QB_NQ.txt', 'w') as f:
    f.write(json.dumps(weight_dict))
  
  # save feature weight  
  prediction_path = 'nqlike_train/nqlike_scores_{}.csv'.format(args.seq)
  if args.predictions:
		# test
    print('Evaluate NQ-like')
    c.save_dictionary(nq_like, prediction_path)
    
