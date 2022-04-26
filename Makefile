all: TriviaQuestion2NQ_Transform_Dataset qanta.train.2018.04.18.json neuralcoref
	
qanta.train.2018.04.18.json: 
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json

TriviaQuestion2NQ_Transform_Dataset:
	wget https://www.dropbox.com/sh/glitdogq6m573f9/AADOjUNbzGZ117UsHtr6K7m5a?dl=1; \
	unzip -d ./TriviaQuestion2NQ_Transform_Dataset/ AADOjUNbzGZ117UsHtr6K7m5a?dl=1; \
	cd TriviaQuestion2NQ_Transform_Dataset; \
	wget https://www.dropbox.com/sh/3xf197ixx30jfey/AAAdC0KNYZl9gPixxR1J7Kx_a?dl=1; \
	unzip -d ./epoch_1_step_26802_NQorig/ AAAdC0KNYZl9gPixxR1J7Kx_a?dl=1; \
	rm -f AAAdC0KNYZl9gPixxR1J7Kx_a?dl=1; \
	pip install -r "requirements.txt"; \
	cd ..; \
	rm -f AADOjUNbzGZ117UsHtr6K7m5a?dl=1; \
	python3 -m nltk.downloader all; \

neuralcoref:
	git clone https://github.com/huggingface/neuralcoref.git; \
	cd neuralcoref; \
	pip install -r requirements.txt; \
	pip install -e .; \

# generate nq_like data and train classifier
generate_frequency: lat_frequency.json

create_nq_like: nq_like_questions.json

generate_classifier: logistic_regression_weight_dict_Qb_NQ.txt

lat_frequency.json: compute_lat_frequency.py qanta.train.2018.04.18.json TriviaQuestion2NQ_Transform_Dataset
	python3 compute_lat_frequency.py; \
	touch compute_lat_frequency.py; \

nq_like_questions.json: transform_question.py TriviaQuestion2NQ_Transform_Dataset neuralcoref
	python3 transform_question.py; \
	touch transform_question.py; \
	
logistic_regression_weight_dict_Qb_NQ.txt: quality_classifier.py TriviaQuestion2NQ_Transform_Dataset
	python3 quality_classifier.py; \
	touch quality_classifier.py; \

# assign nqlike_scores from the classifier to nqlike and rerank the dataset

NQ_plus_NQlike_baseline_train_seq: NQ_NQlike_train_seq_checkpoints

NQ_plus_NQlike_baseline_validation_seq: NQ_NQlike_train_seq_epoch1.txt

# plot EM
EM_from_QA_plotting: plot_nq_nqlike_seq_epoch1.png

	
NQ_NQlike_train_seq_checkpoints: NQ_plus_NQlike_baseline_train_seq.py TriviaQuestion2NQ_Transform_Dataset
	python3 NQ_plus_NQlike_baseline_train_seq.py; \
	touch NQ_plus_NQlike_baseline_train_seq.py; \
	
NQ_NQlike_train_seq_epoch1.txt: NQ_plus_NQlike_baseline_validation_seq.py TriviaQuestion2NQ_Transform_Dataset
	python3 NQ_plus_NQlike_baseline_validation_seq.py; \
	touch NQ_plus_NQlike_baseline_validation_seq.py; \

NQ_NQlike_train_seq_epoch1.png: EM_from_QA_plotting.py NQ_NQlike_train_seq_epoch1.txt
	python3 EM_from_QA_plotting.py; \
	touch EM_from_QA_plotting.py; \

clean:
	#rm -f lat_frequency.json; \
	#rm -f nq_like_questions.json; 
	rm -f logistic_regression_weight_dict_Qb_NQ.txt; \
	rm -f test_feature_dict_QB_NQ.csv; \
	rm -f qanta_train_with_answer_type.json; \
	rm -f qb_with_contexts.json; \
	rm -f nqlike_with_contexts.json; \
	rm -r NQ_NQlike_train_seq_checkpoints; \
	rm -f NQ_NQlike_train_seq_epoch1.txt; \
	#rm -f plot_nq_nqlike_seq_epoch1.png; \
