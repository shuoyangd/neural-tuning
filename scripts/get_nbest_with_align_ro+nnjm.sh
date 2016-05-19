#/bin/sh
set -e
INPUT_FILE="/export/b07/pkoehn/experiment/wmt16-ro-en/tuning/input.tc.2"
CONFIG_FILE="/home/shuoyangd/experiments/wmt16-ro-en/tuning/moses.tuned.ini.2"
OUTPUT_FILE="/home/shuoyangd/neural-tuning/scripts/newsdev2016b.nbest.ro-en.en"
# EMAIL=$2
/home/shuoyangd/mosesstd/bin/moses -config $CONFIG_FILE -mbr -mp -search-algorithm 1 -cube-pruning-pop-limit 5000 -s 5000 -threads 24 -max-trans-opt-per-coverage 100 -t -n-best-list $OUTPUT_FILE 200 distinct -print-alignment-info-in-n-best < $INPUT_FILE > $OUTPUT_FILE.1best

