import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import textstat
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
from collections import Counter
import pickle
import warnings
warnings.filterwarnings("ignore")
import nltk
import argparse


# word2vec library
import gensim.downloader
import gensim.models
from gensim.test.utils import common_texts
# Word2Vec initialization
# w2v_model = gensim.downloader.load('glove-wiki-gigaword-300')

# unigram initialization
MAXFEATURES = 5000
#unigram_converter = CountVectorizer(max_features=MAXFEATURES)




# Step 6: Functions to extract features



def build_dataset(nq, qb, nq_like, limit, percent_test=0.25):
  merged = None
  for label, dataset_location in [(1, nq), (0, qb), (0, nq_like)]:
    print("Reading %s" % dataset_location)
    fold = pd.read_json(dataset_location, lines=True, orient='records')
    if dataset_location == nq_like:
        fold = fold.rename(columns={"quality_score": 'score'})
    if limit >= 0:
      fold = fold.head(limit)

    fold['tokenized'] = fold['question'].apply(lambda x: nltk.tokenize.sent_tokenize(x)[-1])
    fold['label'] = label
    fold['source'] = dataset_location

    if not merged is None:
      merged = pd.concat([merged, fold])
    else:
      merged = fold

  train, test = sklearn.model_selection.train_test_split(merged, train_size=1.0-percent_test, random_state=42)
  print("Labels", train['label'])
  return train, test

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

def count_duplicates(x):
      x_list = x.lower().split(' ')
      counter = Counter(x_list)
      i = 0
      for e in counter:
        if counter[e] > 1:
            i = i+1
      return i

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
  def __init__(self, feature_list, abnormal_length=5, w2v=Word2Vec(), data_length = MAXFEATURES):
    self._length_cutoff = abnormal_length
    self._features = feature_list
    self.w2v = w2v
    self.max_features = data_length
    self.unigram_converter = CountVectorizer(max_features=data_length)
    self.train_dict = None

  def get_ablength(self, df_train):
    return df_train['tokenized'].apply(lambda x: len(x.split()) < self._length_cutoff).values
    
  # function to compute length feature
  def get_length(self, df_train):
    return df_train['tokenized'].apply(lambda x: log(1 + len(x.split()))).values

  #function to compute kincaid readability score
  def get_kincaid(self, df_train):
    return df_train['question'].apply(textstat.flesch_kincaid_grade).values

  # function to compute duplicate feature
  def get_duplicates(self, df_train):    
    return df_train['question'].apply(count_duplicates).values

  # function to compute kincaid readability score
  def get_kincaid(self, df_train):
    return df_train['question'].apply(textstat.flesch_kincaid_grade).values

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

  def get_word2vec(self, df_train):
    return df_train['question'].apply(self.w2v.vectorize).values

  def get_unigram(self, df_train):
    umatrix = []
    if len(df_train) < self.max_features:
      umatrix = self.unigram_converter.transform(df_train['tokenized'])
    else:
      umatrix = self.unigram_converter.fit_transform(df_train['tokenized'])
    unigram_list = self.unigram_converter.inverse_transform(umatrix)
    return unigram_list

  def prepare_features(self, df):
    feature_dictionary = {}
    print("Test data", df)
    y_train = df['score'].values
    # function transform the training data to train
    feature_dictionary['label'] =  y_train    

    for ii in self._features:
      method = getattr(self, "get_" + ii)
      feature_dictionary[ii] = method(df)

    return feature_dictionary

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
    print("XVALS", x_vals)
    return x_vals

  # function to train classifier    
  def train_classifier(self, train):
    self.train_dict = c.prepare_features(train)
    y_train = train['label'].values
    
    print("Training classifer with features: %s" % str(' '.join(self.train_dict.keys())))
    print("YVALS")
    print(y_train)
    x_train = self.get_x_vals(self.train_dict) 
    model = LogisticRegression().fit(x_train,y_train)
    return model

  # function to evaluate classifier
  def evaluate(self, model, df_test):
    test_dict = self.prepare_features(df_test)
    
    x = self.get_x_vals(test_dict)
    results = model.predict_proba(x)
    test_dict['prob_scores'] = results[:, 1]
    predictions = model.predict(x)
    test_dict['pred'] = predictions
    
    # TODO: Compute precision and recall
    #
    # of the things that we say are NQ, how many actually are?
    # of the NQ questions, how many did we find?

    print("PRED", predictions)
    print("LAB", df_test['label'])
    
    acc = accuracy_score(predictions, df_test['label'])
    print('Accuracy: '+format(acc)+'\n')
    return test_dict

  def generate_feature_weight(self, model):
    feature_weight = {}
    weights = model.coef_[0]
    names = list(self.train_dict.keys())
    #assert len(weights) == (len(self._features)), "Unexpected feature length %i vs %i" % (len(self._features) + 1, len(weights))
    for i in range(0,len(names)-1):
      feature_weight[names[i+1]] =[weights[i]] 
    feature_weight["BIAS"] = model.intercept_[0]

    return feature_weight

  def save_dictionary(self, questions, dict_data, file_path):   

    len_of_data = len(questions)
    dict_to_write = {}
    dict_to_write['questions'] = list(questions['tokenized'])
    for col in self._features:
      dict_to_write[col] = dict_data[col]

    dict_to_write['label'] = [binarize(x) for x in dict_data['label']]
    dict_to_write['pred'] = dict_data['pred']
    dict_to_write['source'] = list(questions['source'])
    dict_to_write['good'] = [x > 0 and 'nq_like' in y for x, y in zip(dict_to_write['pred'], dict_to_write['source'])]
    dict_to_write['prob_scores'] = dict_data['prob_scores'] 
    
    df = pd.DataFrame(data=dict_to_write)
    df.to_csv(file_path)


  
# Step 9

# flag = 0 (0 flag refers to wellformedness classifier with an accuracy score) 
# flag = 1 (1 flag refers to NQ-like classifier with 0 accuracy score as we do not have labeled qualityscore for NQ-like)
# qb_last = True: only consider the last sentence of the qb questions
# qb_last = False: consider the whole qb questions
if __name__=="__main__":

  parser = argparse.ArgumentParser(description="Create classifier to discriminate synthetic questions from real questions")
  parser.add_argument('--limit', type=int, default=20)
  parser.add_argument('-f', '--feature_list', nargs='+', default=['length', 'ablength'])
  parser.add_argument('--test_predictions', type=str, default='test_feature_dict_QB_NQ.csv')
  parser.add_argument('--nq_data', type=str, default='./TriviaQuestion2NQ_Transform_Dataset/NaturalQuestions_train_reformatted.json')
  parser.add_argument('--qb_data', type=str, default='./TriviaQuestion2NQ_Transform_Dataset/qb_train_with_contexts_lower_nopunc_debug_Feb24.json')
  parser.add_argument('--nqlike_data', type=str, default='./TriviaQuestion2NQ_Transform_Dataset/nq_like_questions_train_with_orig_quality_scores_with_contexts_Mar10_data_cleaned.json')  
  parser.add_argument('--features', type=str, default='')
  args = parser.parse_args()
	# set flag and if_qb_last_sent here
	# 0 --wellformedness accuracy output
	# 1 --NQ-like output

  train, test = build_dataset(args.nq_data, args.qb_data, args.nqlike_data, args.limit)
  unigram_len = min([len(train), MAXFEATURES])
  c = Classifier(args.feature_list, data_length=unigram_len)


  # Train model and output the weights
  model = c.train_classifier(train)
  weight_dict = c.generate_feature_weight(model)
  print("Weight dict", weight_dict)
  with open('logistic_regression_weight_dict_Qb_NQ.txt', 'w') as f:
    f.write(json.dumps(weight_dict))
  
  # save feature weight  
  if args.test_predictions:
		# test
    print('QB NQ Test ')
    test_dict = c.evaluate(model, test)
    c.save_dictionary(test, test_dict, args.test_predictions)
    
  if args.features:
		# predict nq score for nq-like question
		# transform data
    df_nqlike = pd.read_json(args.nqlike_data, lines=True)
    df_nqlike = df_nqlike[['qanta_id', 'question', 'quality_score']].copy()
    df_nqlike = df_nqlike.rename(columns={"quality_score": 'score'})
		# sample 5% of all questions as dataset too large
    df_nqlike = df_nqlike.sample(frac=0.05, replace=False).reset_index()
    nqlike_feature_dict = c.prepare_features(df_nqlike)
		# predicting and store results
    print('NQ-Like ')
    eval_nqlike = c.evaluate(model, df_nqlike, True)
    c.save_dictionary(df_nqlike['question'],  eval_nqlike, args.features)
