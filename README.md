### Sparse and Low-Tucker Rank Generalized Multi-linear Model (SRT-GMLM)
Repository of experiments on the `SRT-GMLM` model.

Currently working on re-organizing and sharing the experimentation setup and scripts in an easily reproducible way. Please check back soon or feel free to reach out to me through indibimu@msu.edu.


Experiments on Sparse and Low-Tucker Rank Generalized Multi-Linear Model

## Repository Organization

```mermaid
---
config:
    treeView:
        showIcons: true
---

treeView-beta
	SRT-GMLM/
		data/
		srt_gmlm_icassp_exps/
			synthetic/
				configs/    # `.yaml` files specifying the experiment synthetic configurations.
				results/    # Results of experiments and hyper-parameter searches.
				slurm_logs/
				notebooks/
			eeg_adhd/
				notebooks/
					0.00-exploratory_data_analysis-0.ipynb   # Basic Exploratory Data Analysis for the ADHD EEG Dataset.
                    0.01-pre_processing-1.ipynb     # A rundown of the basic pre-processing steps of the dataset.
                    0.02-hyperparameter_search-2.ipynb  # The main script that was used to search for the SRT-GMLM sparsity hyper-parameter.
                    0.03-models_comparison_and_reproduction.ipynb   # Notebook containing the comparison of the models.
		.gitignore
		README.md
		Makefile
		LICENSE
        quickstart.ipynb
```


--------

