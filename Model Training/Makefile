.PHONY: smoke full build-local build-runpod download convert clean-data subset eval-conversion test

MODE ?= smoke

COMPOSE = docker compose -f docker/docker-compose.yml

# --- Smoke / Full training targets ---
smoke: build-local
	$(COMPOSE) run --rm crack-seg-local python src/train.py --mode smoke

full: build-runpod
	docker run --gpus all --rm \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/runs:/workspace/runs \
		crack-seg:runpod python src/train.py --mode full

# --- Docker build ---
build-local:
	docker build -f docker/Dockerfile.local -t crack-seg:local .

build-runpod:
	docker build -f docker/Dockerfile.runpod -t crack-seg:runpod .

# --- Data pipeline ---
download:
	python src/download_data.py --mode $(MODE)

convert:
	python src/convert_annotations.py

clean-data:
	python src/clean_data.py \
		--image-dir data/raw/crackseg9k/images \
		--label-dir data/yolo/labels_raw/crackseg9k \
		--report data/cleaning_report.csv

subset:
	python src/make_subset.py \
		--sources \
		"data/raw/crackseg9k/images:data/yolo/labels_raw/crackseg9k" \
		"data/raw/omnicrack30k/images/training:data/yolo/labels_raw/omnicrack30k/training"

full-subset:
	python src/make_subset.py --all \
		--sources \
		"data/raw/crackseg9k/images:data/yolo/labels_raw/crackseg9k" \
		"data/raw/omnicrack30k/images/training:data/yolo/labels_raw/omnicrack30k/training" \
		"data/raw/omnicrack30k/images/validation:data/yolo/labels_raw/omnicrack30k/validation" \
		"data/raw/omnicrack30k/images/test:data/yolo/labels_raw/omnicrack30k/test"

eval-conversion:
	python src/eval_mask_conversion.py \
		--mask-dir data/raw/crackseg9k/masks \
		--label-dir data/yolo/labels_raw/crackseg9k \
		--n $(N)

test:
	python -m pytest tests/ -v

# --- Full pipeline ---
pipeline: download convert clean-data subset
	@echo "Pipeline ready. Run 'make smoke' to train."
