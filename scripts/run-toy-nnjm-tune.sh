#!/bin/sh
set -e
source $PROJECTDIR/scripts/config.src
python $PROJECTDIR/nnjm_tune.py --target-file $PROJECTDIR/corpus/toy/toy.nbest.trg --alignment-file $PROJECTDIR/corpus/toy/toy.nbest.align --source-file $PROJECTDIR/corpus/toy/toy.src.rep --working-dir $PROJECTDIR/tmp --numberizer-file $PROJECTDIR/corpus/toy/numberizer.pickled
