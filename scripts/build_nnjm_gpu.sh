#!/bin/bash

source /home/shuoyangd/pyenv/theano/bin/activate
cd /home/shuoyangd/neural-tuning
# python nnjm.py --target-file /home/shuoyangd/experiments/wmt16-ro-en/training/corpus.2.en --source-file /home/shuoyangd/experiments/wmt16-ro-en/training/corpus.2.ro --alignment-file /home/shuoyangd/experiments/wmt16-ro-en/model/aligned.2.grow-diag-final-and --working-dir /home/shuoyangd/neural-tuning/ro-en.baseline --vocab-size 16000 # --max-epoch 1
python nnjm.py --target-file /home/shuoyangd/neural-tuning/corpus/toy/toy.1best --source-file /home/shuoyangd/neural-tuning/corpus/toy/toy.src --alignment-file /home/shuoyangd/neural-tuning/corpus/toy/toy.1best.align --working-dir /home/shuoyangd/neural-tuning/ro-en.baseline --vocab-size 10 # --max-epoch 1
