<p align="left">
  <a href="https://github.com/ALS15204/finetune_llm/blob/main/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" target="_blank" />
  </a>
  <a href="https://www.python.org/downloads/release/python-3100/">
    <img alt="License: MIT" src="https://img.shields.io/badge/python-3.10-blue.svg" target="_blank" />
  </a>
</p>

# Finetune LLMs
This repo is developed under Python 3.10

<!-- INSTALLATION -->
## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ALS15204/load_prediction.git
   ```
2. Build venv: under the repo root
   ```sh
   python3 -m venv ./
   ```
3. Install requirements
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- NER TASK -->
## NER task
1. At project root, retrieve data with 
   ```sh
   git clone https://github.com/leslie-huang/UN-named-entity-recognition```
   ```
2. To finetune a model, run
   ```sh
   python scripts/finetune_lm_ner.py` to finetune a model
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>