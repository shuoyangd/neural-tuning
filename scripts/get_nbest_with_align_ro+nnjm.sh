#/bin/sh
set -e
INPUT_FILE="/export/b07/pkoehn/experiment/wmt16-ro-en/tuning/input.tc.2"
CONFIG_FILE="/home/shuoyangd/experiments/wmt16-ro-en/tuning/moses.tuned.ini.2"
OUTPUT_FILE=$1
EMAIL=$2
/home/shuoyangd/mosesstd/scripts/generic/moses-parallel.pl -queue-parameters "-l 'hostname=b*,arch=*64,mem_free=140G,ram_free=140G' -pe smp 20 -M $EMAIL" -decoder /home/shuoyangd/mosesstd/bin/moses -config $CONFIG_FILE -cache-model /mnt/data/$USER/cache -input-file $INPUT_FILE --jobs 4 -decoder-parameters "-mbr -mp -search-algorithm 1 -cube-pruning-pop-limit 5000 -s 5000 -threads 24 -max-trans-opt-per-coverage 100 -t -n-best-list $OUTPUT_FILE 200 distinct -print-alignment-info-in-n-best" > $OUTPUT_FILE.1best

