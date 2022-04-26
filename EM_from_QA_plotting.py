import re
import string
import json
import matplotlib.pyplot as plt
import os.path

def plot(system_scores_path):
  file_exists = os.path.exists(system_scores_path)
  if not file_exists:
      print("Please check if you created the EM scores file from the QA system!")
  with open(system_scores_path) as f:
    scores = f.readlines()
  print("Num of the EM scores: ", len(scores))
  for i in range(len(scores)):
    scores[i] = float(scores[i].strip())

  MrQA1 = []
  for i in range(2000, 26802, 2000):
    MrQA1.append(i)
  MrQA1.append(26802)

  MrQA2 = []
  for i in range(500+26803, 126496+26803, 500):
    MrQA2.append(i)
  samplesMrQA2 = MrQA1 + MrQA2
  samplesMrQA = samplesMrQA2[:-1]

  ## plotting
  plt.figure(figsize=(10, 8))
  plt.plot(samplesMrQA, scores, label = 'NQ+NQlike')

  plt.axvspan(0, 26802,color='g', label='NQ', alpha=0.4)

  plt.title('EM vs Checkpoints for NQ plus NQlike baseline Epoch 1')
  plt.xlabel('Checkpoints')
  plt.ylabel('EM score')
  plt.legend(loc ="upper right")
  plt.show()
  save_path = 'plot_nq_nqlike_seq_epoch1' + '.png'
  plt.savefig(save_path)
  
if __name__ == "__main__":
    system_scores_path = './NQ_NQlike_train_seq_epoch1.txt'
    plot(system_scores_path)