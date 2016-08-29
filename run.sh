#!/bin/sh
set -e
#python nplm.py --training-file ./corpus/correct.tok --working-dir ./tmp 
python nnjm.py --training-file ./corpus/correct.tok --souce-file  ./corpus/ --alignment--working-dir ./tmp 
