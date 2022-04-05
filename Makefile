all: TriviaQuestion2NQ_Transform_Dataset qanta.train.2021.12.20.json neuralcoref 
qanta.train.2021.12.20.json: 
	wget https://obj.umiacs.umd.edu/qanta-jmlr-datasets/qanta.train.2021.12.20.json

lat_frequency.json: compute_lat_frequency.py qanta.train.2021.12.20.json TriviaQuestion2NQ_Transform_Dataset
	python3 compute_lat_frequency.py

nq_like_questions.json: transform_question.py TriviaQuestion2NQ_Transform_Dataset neuralcoref
	python3 transform_question.py

TriviaQuestion2NQ_Transform_Dataset:
	wget https://www.dropbox.com/sh/glitdogq6m573f9/AADOjUNbzGZ117UsHtr6K7m5a?dl=1; \
	unzip -d ./TriviaQuestion2NQ_Transform_Dataset/ AADOjUNbzGZ117UsHtr6K7m5a?dl=1; \
	cd TriviaQuestion2NQ_Transform_Dataset; \
	pip install -r "requirements.txt"; \
	cd ..; \
	rm -f AADOjUNbzGZ117UsHtr6K7m5a?dl=1; \
	python3 -m nltk.downloader all; \

neuralcoref:
	git clone https://github.com/huggingface/neuralcoref.git; \
	cd neuralcoref; \
	pip install -r requirements.txt; \
	pip install -e .; \

clean:
	rm -f qanta.train.2021.12.20.json; \
	rm -r TriviaQuestion2NQ_Transform_Dataset; \
	rm -r neuralcoref; \
