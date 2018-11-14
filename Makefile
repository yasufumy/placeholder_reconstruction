init:
	pip install -r requirements.txt
	./bin/extract_gameid.sh /auto/local/data/corpora/Opta/F13 ./dataset/gameid.txt
	./bin/generate_dataset.py --save-path ./dataset/parallel_5.json
test:
	PYTHONPATH='.':$PYTHONPATH python -m unittest tests/test_*.py
