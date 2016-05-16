#
/bin/sh
set -e
/home/pkoehn/moses/scripts/generic/moses-parallel.pl -queue-parameters "-l 'arch=*64,mem_free=50G,ram_free=50G' -pe smp 12" -decoder /home/pkoehn/moses/bin/moses.2015-03-23 -config /home/pkoehn/experiment/wmt15-fr-en/evaluation/newstest2013.filtered.ini.1 -input-file /home/pkoehn/experiment/wmt15-fr-en/evaluation/newstest2013.input.tc.1 --jobs 5 -decoder-parameters "-mbr -mp -search-algorithm 1 -cube-pruning-pop-limit 5000 -s 5000 -threads 20 -max-trans-opt-per-coverage 100 -t" > tmp

