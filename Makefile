.PHONY: help setup test compile reproduce-deepfakebench reproduce-universal

help:
	@echo "Available targets:"
	@echo "  make setup"
	@echo "  make test"
	@echo "  make compile"
	@echo "  make reproduce-deepfakebench DATA_JSON_FOLDER=... BASE_DATA_PATH=... CHECKPOINT=..."
	@echo "  make reproduce-universal DATA_ROOT=... CHECKPOINT=..."

setup:
	python -m pip install -r requirements.txt
	python -m pip install -r requirements-dev.txt

test:
	python -m pytest -q

compile:
	python -m py_compile train.py test.py data.py model.py losses.py scripts/reproduce.py

reproduce-deepfakebench:
	python scripts/reproduce.py \
		--benchmark deepfakebench \
		--checkpoint "$(CHECKPOINT)" \
		--data-json-folder "$(DATA_JSON_FOLDER)" \
		--base-data-path "$(BASE_DATA_PATH)" \
		$(if $(DATASETS),--datasets "$(DATASETS)") \
		$(if $(DEVICE),--device "$(DEVICE)")

reproduce-universal:
	python scripts/reproduce.py \
		--benchmark universal \
		--checkpoint "$(CHECKPOINT)" \
		--data-root "$(DATA_ROOT)" \
		$(if $(DATASETS),--datasets "$(DATASETS)") \
		$(if $(DEVICE),--device "$(DEVICE)")
