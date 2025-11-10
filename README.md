# PFC-Re-HPC-Model-Codes  
**Repository:** https://github.com/MunenoriTakaku/PFC-Re-HPC-model-codes.git  
**Title of the Study:** *Thalamic regulation improves representation learning in a prefrontal-thalamo-hippocampal network model*

---

## 1. Overview  
This repository contains the original scripts used in the present study submitted to eLife. The study investigates how thalamic regulation contributes to representation learning within a network model combining prefrontal cortex (PFC), thalamus (Re), and hippocampus (HPC).  
All code is provided for transparency and reproducibility. Please note that this is the research-use version of the code, and is not yet optimized for broad public use.

---

## 2. Repository Contents  
- `scripts/` — Core simulation and training scripts for the PFC-Re-HPC network.  
- `notebooks/` — Jupyter notebooks for analysis and visualization (large and environment-dependent).  
- `models/` — Trained model checkpoints (not publicly hosted due to size; see Section 4).  
- `requirements.txt` — Python package dependencies (minimum).  
- `README.md` — This file.

---

## 3. Usage Notes  
- The code was developed under Python 3.9 and PyTorch 1.x; compatibility with other versions is not guaranteed.  
- A GPU with ~10 GB memory or more is recommended for full training.  
- For easier experimentation, a reduced-scale version of the model may be created by adjusting parameters in `scripts/train_model.py` (e.g., lowering hidden units, fewer epochs).  
- Because this is the version used for the study, its structure may appear messy, with limited comments and less modularization than production-ready code.

---

## 4. Data, Notebooks & Models Availability  
Due to the large size of notebooks and trained model weights, they are **not** directly hosted in this repository.  
- Full Jupyter notebooks (with full output) can be made available upon request.  
- Trained model checkpoints are available from the corresponding author upon reasonable request.  
- Where feasible, a cleaned version of the notebooks and model weights will be made publicly available via Zenodo or another data repository upon article acceptance.

---

## 5. Planned Improvements  
After the manuscript is accepted and published, a cleaned and documented version of the code will be released. It will include:
- A reorganized directory structure  
- Enhanced comments and docstrings  
- Simplified example scripts for demonstration purposes  
- Possibly a Docker / Conda environment file for easier reproducibility

---

## 6. Contact & License  
Maintained by Munenori Takaku.  
Contact: [email address or GitHub handle]  
License: MIT License (see `LICENSE` file)  
Feel free to fork the repository and adapt it for your research, provided proper citation of the original study.

---

## 7. Citation  
If you use the code or rely on the results, please cite:  
> Takaku, M., *Thalamic regulation improves representation learning in a prefrontal-thalamo-hippocampal network model*. (Submitted to eLife)

---

Thank you for your interest in this work!  
