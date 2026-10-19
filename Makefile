#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = SRT-GMLM
PYTHON_VERSION = 3.13.12
PYTHON_INTERPRETER = python

UV_RUN = uv run
VENV = .venv
PYTHON = $(VENV)/bin/python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.venv: pyproject.toml uv.lock
	uv sync
	@touch .venv

## Install Python dependencies
.PHONY: requirements
requirements: .venv
	uv sync
	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 srt_gmlm_icassp_exps
	isort --check --diff srt_gmlm_icassp_exps
	black --check srt_gmlm_icassp_exps

## Format source code with black
.PHONY: format
format:
	isort srt_gmlm_icassp_exps
	black srt_gmlm_icassp_exps



## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

EEG_ADHD_DATASET_ZIP = ./data/eeg_adhd/eeg-dataset-for-adhd.zip
$(EEG_ADHD_DATASET_ZIP):
	mkdir -p $(dir $@)
	curl -L -o $@ https://www.kaggle.com/api/v1/datasets/download/danizo/eeg-dataset-for-adhd

EEG_ADHD_DATASET_CSV = ./data/eeg_adhd/adhdata.csv
$(EEG_ADHD_DATASET_CSV): $(EEG_ADHD_DATASET_ZIP)
	mkdir -p $(dir $@)
	unzip $(EEG_ADHD_DATASET_ZIP) -d $(dir $@)

## Download the EEG Dataset for ADHD from Kaggle with cURL and extract as a csv file.
.PHONY: download_eeg_adhd
download_eeg_adhd: $(EEG_ADHD_DATASET_CSV)
	rm $(EEG_ADHD_DATASET_ZIP)


## Parse EEG ADHD dataset from csv format to .fif RAW formatted EEG readings
.PHONY: parse_eeg_adhd_csv_to_fif
parse_eeg_adhd_csv_to_fif: .venv $(EEG_ADHD_DATASET_CSV)
	$(UV_RUN) python ./srt_gmlm_icassp_exps/eeg_adhd/adhd_mne.py --save-fif


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) srt_gmlm_icassp_exps/dataset.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
