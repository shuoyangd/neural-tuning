#!/bin/sh
set -e
python nplm_adadelta.py --training-file ./corpus/correct.tok --working-dir ./tmp 
