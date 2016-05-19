#/bin/sh
set -e
INPUT_FILE="/home/shuoyangd/experiments/wmt16-ru-en/tuning/input.tc.1"
CONFIG_FILE="/home/shuoyangd/experiments/wmt16-ru-en/tuning/moses.tuned.ini.1"
OUTPUT_FILE="/home/shuoyangd/neural-tuning/scripts/newstest2015.nbest.ru-en.en"
# EMAIL=$2
/home/shuoyangd/mosesstd/bin/moses -config $CONFIG_FILE -mbr -mp -search-algorithm 1 -cube-pruning-pop-limit 5000 -s 5000 -threads 24 -max-trans-opt-per-coverage 100 -t -n-best-list $OUTPUT_FILE 200 distinct -print-alignment-info-in-n-best < $INPUT_FILE > $OUTPUT_FILE.1best

