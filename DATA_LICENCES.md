# Data and licence notes (checked November 2026; re-check before reuse)

| Dataset | Used in | Terms as found | Action |
|---|---|---|---|
| PneumoniaMNIST (MedMNIST+) | NB04-NB07, NB09, NB10, X2, X3 | CC BY 4.0 (MedMNIST v2 documentation); `medmnist` package 3.0.2 is Apache-2.0 | Downloaded by the notebooks; cite Yang et al., Scientific Data 2023 |
| Montgomery County CXR set (NLM) | NB08, NB-X1 | 138 images (80 normal, 58 TB); the dataset ReadMe asks users not to share; no licence is named | **Not in the repo, no link in the notebooks.** Students get a derived 256 px `.npz` from Moodle (enrolled students only); the instructor builds it with `scripts/prepare_montgomery_subset.py` from the official NLM download |
| Kvasir-SEG | (mentioned in slides only) | Research and educational use; commercial use forbidden without written permission (Simula) | Not used in any notebook |
| Synthetic arrays / digits | NB00-NB03, NB-B | Generated in code / scikit-learn bundled data | None |

**Not verified:** the exact redistribution rights of Montgomery for course use (ask NLM if the course will be repeated or opened); pydicom and nibabel version pins in `requirements.txt` (written from memory of the release lines, not checked against PyPI).

Repository code licence: **TBD by the instructor** (no `LICENSE` file has been added on purpose).
