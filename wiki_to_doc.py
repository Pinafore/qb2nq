from typing import List
import requests
import pandas as pd
from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import RAGenerator, DensePassageRetriever
from haystack.utils import print_answers, fetch_archive_from_http


def tutorial7_rag_generator():
    # Add documents from which you want generate answers
    # Download a csv containing some sample documents data
    # Here some sample documents data
    #doc_dir = "data/tutorial7/"
    #s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/small_generator_dataset.csv.zip"
    # fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # Get dataframe with columns "title", and "text"
    df = pd.read_csv("psgs_w100_tmp.tsv", sep="\t")
    # Minimal cleaning
    df.fillna(value="", inplace=True)

    print(df.head())

    titles = list(df["title"].values)
    texts = list(df["text"].values)

    # Create to haystack document format
    documents: List[Document] = []
    for title, text in zip(titles, texts):
        documents.append(Document(content=text, meta={"name": title or ""}))

    # Initialize FAISS document store to documents and corresponding index for embeddings
    # Set `return_embedding` to `True`, so generator doesn't have to perform re-embedding
    # Don't forget to install FAISS dependencies with `pip install farm-haystack[faiss]`
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)

    # Initialize DPR Retriever to encode documents, encode question and query documents
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=True,
        embed_title=True,
    )

    # Initialize RAG Generator
    generator = RAGenerator(
        model_name_or_path="facebook/rag-token-nq",
        use_gpu=True,
        top_k=1,
        max_length=200,
        min_length=2,
        embed_title=True,
        num_beams=2,
    )

    # Delete existing documents in documents store
    document_store.delete_documents()
    # Write documents to document store
    document_store.write_documents(documents)
    # Add documents embeddings to index
    document_store.update_embeddings(retriever=retriever)



if __name__ == "__main__":
    tutorial7_rag_generator()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/