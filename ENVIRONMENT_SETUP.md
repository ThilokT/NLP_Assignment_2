# Environment Setup Guide

This guide details the step-by-step instructions to set up the environment for **Project 7 (HaluEval QA) - NLP Assignment 2**.

---

## 1. Prerequisites & Server Context

- **Server GPU:** 2 × NVIDIA Tesla K80 (`sm_37` Kepler architecture, Driver 470, CUDA 11.4).
- **Target Framework:** `torch==2.4.1`, `transformers==4.44.2`.
- **Target OS/Python:** Linux, Python 3.10 – 3.12.

> [!NOTE]
> **CPU vs. GPU Note:**
> - **Data Setup & Task 3** run on **CPU** and will execute smoothly on this server.
> - `torch==2.4.1` dropped support for Kepler GPUs (`Tesla K80`), so running modern PyTorch on GPU requires compute capability ≥ 5.0 (e.g., Turing T4 / Ampere / Ada / Hopper).
> - For heavy GPU inference in **Tasks 1 and 2**, use **Google Colab (T4 GPU)** or run in CPU mode if model size permits.

---

## 2. Setup Instructions

### Step 1: Navigate to the Project Directory
```bash
cd ~/23CS30073_NLP_Assignment_2
```

### Step 2: Create & Activate Virtual Environment
We recommend using `uv` for fast package management:

```bash
# 1. Create a virtual environment using Python 3.10
uv venv --python 3.10 .venv

# 2. Activate the virtual environment
source .venv/bin/activate
```

> **Tip:** In any future terminal session, re-activate the virtual environment with:
> ```bash
> source ~/23CS30073_NLP_Assignment_2/.venv/bin/activate
> ```

---

### Step 3: Install Required Dependencies

Install the exact package versions specified in `requirements.txt`:
```bash
uv pip install -r requirements.txt
```

If you plan to run Jupyter Notebooks (`.ipynb`), install the notebook kernel:
```bash
uv pip install ipykernel jupyter
python -m ipykernel install --user --name=nlp-assign2 --display-name="Python (NLP Assign 2)"
```

---

### Step 4: Configure Hugging Face Token (if applicable)
If your tasks require downloading gated Hugging Face models/datasets, authenticate with the token stored in `token.txt`:
```bash
export HF_TOKEN=$(cat token.txt)
# Or log in interactively:
# uv run huggingface-cli login --token $(cat token.txt)
```

---

### Step 5: Verify the Environment
Run this quick check to confirm all packages are loaded properly:
```bash
python -c "import torch, transformers, datasets, sklearn; print('PyTorch version:', torch.__version__); print('Transformers version:', transformers.__version__); print('Datasets version:', datasets.__version__); print('Environment successfully verified.')"
```

