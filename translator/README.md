## CSV Translator (M2M100 + GPU)

This script translates selected columns in a CSV file into English using the M2M100 multilingual model.  
It supports automatic language detection and GPU acceleration.

---

### Requirements

```bash
pip install pandas torch transformers langid
```

If you have a GPU, install the CUDA version of PyTorch *(CUDA 12.1 recommended)*:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### Model Setup

Download the model from Hugging Face:

```bash
git lfs install
git clone https://huggingface.co/facebook/m2m100_418M models/m2m100_418M
```

You can also use it directly by setting:

```bash
--model facebook/m2m100_418M
```

By default, the script expects the model at: `models/m2m100_418M`

Please organize your folder in this way, or **specify a different model path** when running the script:

```bash
--model facebook/m2m100_418M
```

---

### Usage

```bash
python csv_translator_m2m100_gpu.py \
  --csv data/input.csv \
  --columns DESCRIPTION ACTION_TAKEN \
  --out outputs/translated.csv \
  --model models/m2m100_418M \
  --target en
```

---

### Arguments

| Argument    | Description                          | Required | Default              |
| ----------- | ------------------------------------ | -------- | -------------------- |
| `--csv`     | Path to input CSV file               | Yes      | —                    |
| `--columns` | List of column names to translate    | Yes      | —                    |
| `--out`     | Path to save the translated CSV file | Yes      | —                    |
| `--model`   | Model name or local path             | No       | `models/m2m100_418M` |
| `--target`  | Target language code                 | No       | `en` (English)       |

---

### Running on a Cluster (SLURM)

You can submit the translation job with `sbatch`.
Example script `run_translate.sbatch` is in the folder.

Submit the job with:

```bash
sbatch run_translate.sbatch
```
