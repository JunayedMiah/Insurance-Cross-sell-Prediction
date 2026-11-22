.PHONY: install sample train serve test lint docker

install:
	pip install -r requirements-dev.txt

sample:
	python scripts/make_sample_data.py --rows 20000 --out data/sample.csv

train:
	PYTHONPATH=src python -m insurance_api.train --data $${DATA:-data/train.csv} --out models/model.joblib

serve:
	PYTHONPATH=src uvicorn insurance_api.main:app --reload

test:
	pytest -q

lint:
	ruff check src tests scripts

docker:
	docker compose up --build
