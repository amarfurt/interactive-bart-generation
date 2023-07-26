# interactive-bart-generation
Interactive generation with BART.

## Setup
First, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) for your operating system.

Then, run the following in a terminal to install the required libraries:
```
cd <path to this folder>
conda create -n intrepid python=3.9
conda activate intrepid
pip install -r requirements.txt
conda deactivate
```

## Usage
Once you have set up the project, you can enter interactive generation with:
```
cd <path to this folder>
conda activate intrepid
python run_bart.py
```
