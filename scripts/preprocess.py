#! /usr/bin/python

# Feed me the source file and a n-best list with alignment as generated by moses decoder
# I'll spit out repeated source (,rep), 
#   index of repeated source (.map),
#   target sentence (.trg) and alignment (.align) in it
# 
# May, 2016 @ Notre Dame

import sys
import re

if len(sys.argv) <= 2:
  sys.stderr.write("usage: python preprocess.sh SOURCE_INPUT NBEST_LIST_INPUT\n")
  sys.exit(1)

source_file = open(sys.argv[1])
source_rep_file = open(sys.argv[1] + ".rep" , 'w')
source_map_file = open(sys.argv[1] + ".map", 'w')
nbest_list_file = open(sys.argv[2])
target_file = open(sys.argv[2] + ".trg", 'w')
alignment_file = open(sys.argv[2] + ".align", 'w')

source_id = 0
source_line = source_file.readline()
for nbest_line in nbest_list_file:
  nbest_fields = nbest_line.split(" ||| ")
  if int(nbest_fields[0]) != source_id:
    source_id = int(nbest_fields[0])
    source_line = source_file.readline()
  source_rep_file.write(source_line)
  source_map_file.write(str(source_id) + '\n')

  # extract target tokens
  raw_trg_tokens = re.findall("\\| [^|]+ \\|", "| " + nbest_fields[1])
  trg_tokens = []
  for raw_trg_token in raw_trg_tokens:
    trg_token = raw_trg_token[1 : len(raw_trg_token) - 1]
    trg_tokens.append(trg_token.strip())
  target_file.write(" ".join(trg_tokens) + "\n") 

  alignment_file.write(nbest_fields[-1])

alignment_file.close()
target_file.close()
nbest_list_file.close()
source_map_file.close()
source_rep_file.close()
source_file.close()
