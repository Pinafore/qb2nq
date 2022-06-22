# Here are some imports that we'll need

from haystack.nodes import DensePassageRetriever
from haystack.utils import fetch_archive_from_http
from haystack.document_stores import InMemoryDocumentStore

# Here are the variables to specify our training data, the models that we use to initialize DPR
# and the directory where we'll be saving the model
doc_dir = "./"
train_filename = "nqdpr.json"
dev_filename = "nqdpr.json"

query_model = "bert-base-uncased"
passage_model = "bert-base-uncased"

save_dir = "../saved_models/dpr"

## Initialize DPR model

retriever = DensePassageRetriever(
    document_store=InMemoryDocumentStore(),
    query_embedding_model=query_model,
    passage_embedding_model=passage_model,
    max_seq_len_query=64,
    max_seq_len_passage=256,
)

# Start training our model and save it when it is finished

retriever.train(
    data_dir=doc_dir,
    train_filename=train_filename,
    dev_filename=dev_filename,
    test_filename=dev_filename,
    n_epochs=1,
    batch_size=16,
    grad_acc_steps=8,
    save_dir=save_dir,
    evaluate_every=3000,
    embed_title=False,
    num_positives=1,
    num_hard_negatives=30,
)