import os
import re
#os.environ['TRANSFORMERS_CACHE'] = '/fs/clip-scratch/tasnim/new_fol_temp/temp_cache'
#os.environ['HF_DATASETS_CACHE'] = '/fs/clip-scratch/tasnim/new_fol_temp/temp_cache'
import json
f = open('/fs/clip-scratch/tasnim/hay2/DPR/DPR/dpr/downloads/data/retriever/nq-dev.json')
#f1 = open('nq_like_squad.json','w')
data = json.load(f)
#print(type(data))
from datasets import load_dataset
wiki = load_dataset("wiki_dpr", with_embeddings=False, with_index=True, split="train")

from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder 
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base') 
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base') 
for i in data:
    found=False
    question = i["question"]
    print(question)
    answer=i['answers']
    #answer1=re.sub("\(.*?\)|\[.*?\]","",answer)
    #answer1=answer1.replace("{","").replace("}","")
    i["positive_ctxs"]=[]
    i["negative_ctxs"]=[]
    i["hard_negative_ctxs"]=[]
    #print(answer1)
    #"answer": "{Drosophila} melanogaster [or {fruit} fly; or {D.} melanogaster]", "page": "Drosophila_melanogaster"
    question_emb = question_encoder(**question_tokenizer(question, return_tensors="pt"))[0].detach().numpy() 

    passages_scores, passages = wiki.get_nearest_examples("embeddings", question_emb, k=1000) # get k nearest neighbors
    #wiki.get(title="Drosophila_melanogaster")
    title={}
    passage={}
    id_={}
    for k, v in passages.items():
        if k=="text":
            passage=v
        if k=="title":
            title=v
        if k=="id":
            id_=v
    """print(passages)
    print(passages_scores)
    print(title,type(title))
    print(id_,type(id_))
    print(passage,type(passage))"""
    #print(i[])
    #print(type(i))
    count=0
    for k, v in passages.items():
        #print("v",v,"k",k)
        if k=="text":
            for l in v:
                #if l.lower().find(answer.any().lower())!=-1:
                if any(m.lower() in l.lower() for m in answer):
                    a={'title': title[count], 'text': passage[count], 'score': str(int(passages_scores[count])), 'title_score': str(0), 'passage_id': str(id_[count])}
                    i["positive_ctxs"].append(a)
                    found=True
                else:
                    if found==False:
                        a={'title': title[count], 'text': passage[count], 'score': str(int(passages_scores[count])), 'title_score': str(0), 'passage_id': str(id_[count])}
                        i["hard_negative_ctxs"].append(a)
                    else:
                        a={'title': title[count], 'text': passage[count], 'score': str(int(passages_scores[count])), 'title_score': str(0), 'passage_id': str(id_[count])}
                        i["negative_ctxs"].append(a)
                count=count+1
    #print("After ",type(i))
    with open('/fs/clip-scratch/tasnim/hay2/DPR/DPR/dpr/downloads/data/retriever/nq-dev_un.json', 'a') as outfile:
        json.dump(i,outfile,indent=4,  
                        separators=(',',': '))


