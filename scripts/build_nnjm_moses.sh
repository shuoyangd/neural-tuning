#/bin/sh
set -e
WORKING_DIR=$1
# /home/shuoyangd/mosesstd/scripts/training/bilingual-lm/train_nplm.py -w /home/shuoyangd/experiments/wmt16-en-ro/lm/nnjm.blm.6 -c /home/shuoyangd/experiments/wmt16-en-ro/training/corpus.2 -r /home/shuoyangd/experiments/wmt16-en-ro/lm/nnjm.blm.6 -l /home/shuoyangd/nplm/ -n 14 -e 10 --threads 30
#/home/shuoyangd/mosesstd/scripts/training/bilingual-lm/train_nplm.py -w /home/shuoyangd/experiments/wmt16-en-ro/lm/nnjm.blm.6 -c /home/shuoyangd/experiments/wmt16-en-ro/training/corpus.2 -r /home/shuoyangd/experiments/wmt16-en-ro/lm/nnjm.blm.6 -l /home/shuoyangd/nplm/ -n 14 -e 1
/home/shuoyangd/mosesstd/scripts/training/bilingual-lm/train_nplm.py -w $WORKING_DIR -c /home/arenduc1/Projects/neural-tuning/corpus/nnjm-tiny-corpus/corpus.2.10 -r $WORKING_DIR -l /home/shuoyangd/nplm/ -n 14 -e 1
