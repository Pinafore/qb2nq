import json
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import FAISSDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http, print_answers
from haystack.nodes import FARMReader, TransformersReader
from typing import Dict, Any, List, Optional

import json
import pprint
import logging
from collections import defaultdict

import pandas as pd

from haystack.schema import Document, Answer
from haystack.document_stores.sql import DocumentORM

def format_answers(results: dict, details: str = "all", max_text_len: Optional[int] = None):
    """
    Utility function to print results of Haystack pipelines
    :param results: Results from a pipeline
    :param details: One of "minimum", "medium", "all". Defining the level of details to print.
    :param max_text_lenght: shorten lengthy text fields to the maximum allowed length. Set to
        None to not cut long text.
    :return: None
    """
    # Defines the fields to keep in the Answer for each detail level
    fields_to_keep_by_level = {"minimum": ["answer", "context"], "medium": ["answer", "context", "score"]}

    if not "answers" in results.keys():
        raise ValueError(
            "The results object does not seem to come from a Reader: "
            f"it does not contain the 'answers' key, but only: {results.keys()}.  "
            "Try print_documents or print_questions."
        )

    if "query" in results.keys():
        print(f"\nQuery: {results['query']}\nAnswers:")

    answers = results["answers"]
    pp = pprint.PrettyPrinter(indent=4)

    # Filter the results by detail level
    filtered_answers = []
    if details in fields_to_keep_by_level.keys():
        for ans in answers:
            filtered_ans = {
                field: getattr(ans, field)
                for field in fields_to_keep_by_level[details]
                if getattr(ans, field) is not None
            }
            filtered_answers.append(filtered_ans)
    elif details == "all":
        filtered_answers = answers
    else:
        valid_values = ", ".join(fields_to_keep_by_level.keys()) + " and 'all'"
        logging.warn(f"print_answers received details='{details}', which was not understood. ")
        logging.warn(f"Valid values are {valid_values}. Using 'all'.")
        filtered_answers = answers

    # Shorten long text fields
    if max_text_len is not None:
        for ans in answers:
            if getattr(ans, "context") and len(ans.context) > max_text_len:
                ans.context = ans.context[:max_text_len] + "..."

    return filtered_answers

def pos_context(str):
    prediction = pipe.run(
    query=str, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 1}}
    )
    print_answers(prediction, details="minimum")
    prediction=format_answers(prediction, details="minimum")
    return prediction

document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
doc_dir = "wiki"
#s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt6.zip"
#fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
    
# Convert files to dicts
docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
document_store.write_documents(docs)
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    max_seq_len_query=64,
    max_seq_len_passage=256,
    batch_size=16,
    use_gpu=True,
    embed_title=True,
    use_fast_tokenizers=True,
    )
    # Important:
    # Now that after we have the DPR initialized, we need to call update_embeddings() to iterate over all
    # previously indexed documents and update their embedding representation.
    # While this can be a time consuming operation (depending on corpus size), it only needs to be done once.
    # At query time, we only need to embed the query and compare it the existing doc embeddings which is very fast.
document_store.update_embeddings(retriever)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
pipe = ExtractiveQAPipeline(reader, retriever)
with open('nqlike_train10.json') as json_file:
    data = json.load(json_file)
    print(data)
    
for i in data:
    print("hello")
    prediction=pos_context(i["question"])
    print(prediction,"type: ",type(prediction))
    for j in prediction:
        i["context"]=j["context"]
    print(i["question"])
    
import json
with open('nqlike_train10_p.json', 'w') as f:
    json.dump(data, f)