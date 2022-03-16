
qanta.train.2021.12.20.json: 
	wget https://obj.umiacs.umd.edu/qanta-jmlr-datasets/qanta.train.2021.12.20.json

qanta_train_with_answer_type_v1.json: compute_lat_frequency.py qanta.train.2021.12.20.json
	python3 compute_lat_frequency.py
