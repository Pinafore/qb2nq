import json
import csv
 
with open('/fs/clip-scratch/tasnim/hay2/DPR/DPR/dpr/downloads/data/retriever/trainall.json') as json_file:
    jsondata = json.load(json_file)
with open('/fs/clip-scratch/tasnim/hay2/DPR/DPR/train/downloads/data/retriever/nq-train.json') as json_file:
    jsondata2 = json.load(json_file)
b=open('/fs/clip-scratch/tasnim/hay2/DPR/DPR/dpr/downloads/data/retriever/train_tmp.json', 'w')
for i in jsondata:
    #print(i)
    ans=[]
    #if i[]
    ans.append(i['answer'])
    x = {
        "question_col": i['question'],
        "answers_col": ans
        }
    with open('/fs/clip-scratch/tasnim/hay2/DPR/DPR/dpr/downloads/data/retriever/train_tmp.json', 'a') as outfile:
        json.dump(x, outfile, indent=2)

for i in jsondata2:
    #print(i)
    x = {
        "question_col": i['question'],
        "answers_col": i['answers']
        }
    with open('/fs/clip-scratch/tasnim/hay2/DPR/DPR/dpr/downloads/data/retriever/train_tmp.json', 'a') as outfile:
        json.dump(x, outfile, indent=2)
