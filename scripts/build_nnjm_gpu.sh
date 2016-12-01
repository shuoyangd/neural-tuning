#!/bin/bash
#$ -q g.q
#$ -M dings@jhu.edu
#$ -l 'arch=*64,gpu=1'
#$ -o /home/shuoyangd/neural-tuning/scripts/logs -e /home/shuoyangd/neural-tuning/scripts/logs

source /home/shuoyangd/pyenv/theano/bin/activate
cd /home/shuoyangd/neural-tuning
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu,on_unused_input=warn  python nnjm.py --target-file /home/shuoyangd/experiments/wmt16-ro-en/training/corpus.2.en --source-file /home/shuoyangd/experiments/wmt16-ro-en/training/corpus.2.ro --alignment-file /home/shuoyangd/experiments/wmt16-ro-en/model/aligned.2.grow-diag-final-and --working-dir /home/shuoyangd/neural-tuning/ro-en.baseline --vocab-size 50000 -vtf /home/shuoyangd/neural-tuning/data/val.trg.500 -vsf /home/shuoyangd/neural-tuning/data/val.src.500 -vaf /home/shuoyangd/neural-tuning/data/val.align.500 --max-epoch 10
# python nnjm.py --target-file /home/shuoyangd/neural-tuning/corpus/toy/toy.1best --source-file /home/shuoyangd/neural-tuning/corpus/toy/toy.src --alignment-file /home/shuoyangd/neural-tuning/corpus/toy/toy.1best.align --working-dir /home/shuoyangd/neural-tuning/ro-en.baseline --vocab-size 40 # --max-epoch 1
