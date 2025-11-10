# PFC-Re-HPC-Model-Codes  
**Repository:** https://github.com/MunenoriTakaku/PFC-Re-HPC-model-codes.git  
**Title of the Study:** *Thalamic regulation improves representation learning in a prefrontal-thalamo-hippocampal network model*

---

## 1. Overview  
This repository contains the original scripts used in the present study submitted to eLife. The study investigates how thalamic regulation contributes to representation learning within a network model combining prefrontal cortex (PFC), thalamus (Re), and hippocampus (HPC).  
All code is provided for transparency and reproducibility. Please note that this is the research-use version of the code, and is not yet optimized for broad public use.

---

## 2. Repository Contents  

All files are provided in a single directory for convenience.  
They can be broadly categorized into three functional groups: **training scripts**, **analysis notebooks**, and **visualization notebooks**.

### Model Training Scripts
- **`PFC_HPC_Thalamo_*.py`**  
  Training scripts for the **three-module (PFC–Re–HPC)** network model.  
  Each file corresponds to a specific behavioral task variant (three in total).  

- **`v4_3_*.py`**  
  Training scripts for the **two-module (PFC–HPC)** model, used for comparison and ablation studies.  

### Analysis Notebooks
- **`Testing_note_*.ipynb`**  
  Jupyter notebooks for post-training analysis of network activity and representation structure.  
  Two versions are provided: one for the **standard task** and one for the **H-task** variant.
  Note that these notebooks require pre-trained model outputs and may not execute directly without access to trained weights.


### Visualization Notebooks
- **`FigMake_*.ipynb`**  
  Jupyter notebooks for figure generation and visualization.  
  They combine processed analysis outputs to reproduce the key plots shown in the manuscript (e.g., PCA embeddings, trial trajectories).  
  Note that these notebooks require pre-trained model outputs and may not execute directly without access to trained weights.

---

**Note:**  
All scripts were used as part of the actual computational experiments reported in the study.  
They are provided *as-is* for transparency and reproducibility, and may not be fully modular or optimized for external execution.

---

## 3. Usage Notes  
- The code was developed under Python 3.9 and PyTorch 2.0; compatibility with other versions is not guaranteed.  
- Because this is the version used for the study, its structure may appear messy, with limited comments and less modularization than production-ready code.

---

## 4. Data, Notebooks & Models Availability  
Due to the large size of trained model weights, they are **not** directly hosted in this repository.  
- Trained model checkpoints are available from the corresponding author upon reasonable request.  
- Where feasible, a cleaned version of the notebooks and model weights will be made publicly available via Zenodo or another data repository upon article acceptance.

---

## 5. Planned Improvements  
A cleaned and documented version of the code will be prepared.  
The planned updates include:  
- A reorganized directory structure  
- Enhanced comments and docstrings
- Refactoring key components (model, training, analysis) into importable Python modules  


---

Thank you for your interest in this work!  
