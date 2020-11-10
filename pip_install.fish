#!/bin/fish

source venv/bin/activate.fish

pip install --upgrade pip

pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install transformers==3.0.2 pytorch_lightning lz4==3.1.0 rouge-score datasets nlp scikit-learn tensorboard spacy ipython torch_optimizer

