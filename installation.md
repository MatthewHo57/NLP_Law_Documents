# Installation Guide

This guide will help you set up the Legal Document Analyzer project from scratch, resolving common dependency issues.

## Step 1: Create a Virtual Environment (Recommended)

Creating a virtual environment helps isolate project dependencies from your system Python:

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

## Step 2: Install Dependencies

Install all dependencies using pip:

```bash
# Make sure pip is up to date
python -m pip install --upgrade pip

# Install all requirements
python -m pip install -r requirements.txt
```

## Step 3: Download the spaCy Language Model

After installing spaCy, download the English language model:

```bash
python -m spacy download en_core_web_sm
```

## Troubleshooting Common Issues

If you encounter errors installing packages, try these solutions:

### "No module named 'X'" Errors

Install individual packages that are causing issues:

```bash
# For spaCy
python -m pip install spacy

# For pdfminer
python -m pip install pdfminer.six

# For scikit-learn
python -m pip install scikit-learn

# For torch (PyTorch)
python -m pip install torch
```

### PyTorch Installation Issues

If you have trouble with PyTorch, install it separately following instructions from the [PyTorch website](https://pytorch.org/get-started/locally/):

```bash
# CPU-only version (simpler)
python -m pip install torch torchvision torchaudio
```

### Transformers Installation Issues

If transformers installation fails:

```bash
python -m pip install --upgrade transformers
```

## Verifying Installation

To verify that everything is installed correctly, run:

```bash
python -c "import spacy, torch, sklearn, transformers, pdfminer, summa, datasets, matplotlib; print('All dependencies successfully imported!')"
```

## Running the Demo

After successful installation, you can run the demo:

```bash
python demo.py --pdf path/to/your/document.pdf
```

If you don't have a PDF ready, create a sample text file and use it instead (modify the code slightly to accept .txt files).