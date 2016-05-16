# rnlm -- a theano-based recurrent neural network language model
# 
# proudly developed by
# Shuoyang Ding @ Johns Hopkins University
# 
# with the help from this blog post:
# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
# 
# March, 2016

import argparse
from thibetanus.basic.indexer import indexer
import logging
import numpy as np
import pickle
from rnn import rnn
import sys

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--training-file", "-t", dest="training_file", metavar="PATH", help="File used as training corpus.", required=True)
parser.add_argument("--working-dir", "-w", dest="working_dir", metavar="PATH", help="Directory used to dump models etc.", required=True)
parser.add_argument("--learning-rate", dest="learning_rate", type=float, metavar="FLOAT", help="Learning rate used to update weights (default = 0.25).")
parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, metavar="INT", help="Dimension of the hidden layer (default = 100).")
parser.add_argument("--bptt-truncate", dest="bptt_truncate", type=int, metavar="INT", help="Maximum bptt level (default = 4).")
# parser.add_argument("--minibatch-size", "-b", dest="minibatch_size", type=int, metavar="INT", help="Minibatch size (in sentences) of SGD.")
parser.add_argument("--gradient-check", dest="gradient_check", type=int, metavar="INT", help="The iteration interval for gradient check. Pass 0 if gradient check should not be performed (default = 0).")
parser.add_argument("--save-interval", dest="save_interval", type=int, metavar="INT", help="The epoch interval for saving models. Pass 0 if wish to save only once at the end of each epoch (default = 0).")
parser.add_argument("--max-epoch", dest="max_epoch", type=int, metavar="INT", help="Maximum number of epochs should be performed during training (default = 5).")

parser.set_defaults(
  learning_rate=0.25,
  hidden_dim=100,
  bptt_truncate=4,
  minibatch_size=100,
  gradient_check=0,
  save_interval=0,
  max_epoch=5)

def main(options):

  # collecting vocab
  logging.info("start collecting vocabulary")
  training_corpus = open(options.training_file)
  # TODO: for machines with larger memories, should use memory mapping instead of disk mapping
  indexed_corpus = open(options.working_dir + "/indexed_corpus", 'w')
  vocab = indexer()
  vocab.add("</s>")
  for sentence in training_corpus:
    tokens = ["<s>"]
    tokens.extend(sentence.strip().split(' '))
    indexed_sentence = ""
    for token in tokens:
      ix = vocab.getIndex(token)
      indexed_sentence += (str(ix) + " ")
    indexed_corpus.write(indexed_sentence.strip() + "\n")
  logging.info("vocabulary collection finished")
  indexed_corpus.close()
  training_corpus.close()

  # training
  logging.info("start training with vocabulary size {0}, learning rate {1}, hidden dimension {2}, bptt truncate {3}"
      .format(vocab.size(), options.learning_rate, options.hidden_dim, options.bptt_truncate))
  net = rnn(vocab.size(), options.hidden_dim, options.bptt_truncate)
  for epoch in range(options.max_epoch):
    logging.info("epoch {0} started".format(epoch))
    indexed_corpus = open(options.working_dir + "/indexed_corpus")
    instance_count = 0
    total_loss = 0.0
    for sentence in indexed_corpus:
      tokens = sentence.split(' ')
      x = [int(token) for token in tokens]
      y = x[1:]
      y.append(vocab.indexOf("</s>"))
      net.sgd(x, y, options.learning_rate)
      total_loss += net.loss(x, y)
      instance_count += 1
      if instance_count % 1 == 0:
        logging.info("{0} instances seen".format(instance_count))
      if options.gradient_check != 0 and instance_count % options.gradient_check == 0:
        net.gradient_check(x, y)
      if options.save_interval != 0 and instance_count % options.save_interval:
        dump_file = open(options.working_dir + "/net." + str(instance_count))
        pickle.dump(net, dump_file)
        dump_file.close()
      # print net.U.get_value()
      # print net.V.get_value()
      # print net.W.get_value()
    indexed_corpus.close()
    logging.info("epoch {0} finished with loss {1}".format(epoch, total_loss))
    # if options.save_interval == 0:
      # dump_file = open(options.working_dir + "/net." + str(epoch), 'w')
      # pickle.dump(net, dump_file)
      # dump_file.close()
  logging.info("trainining finished with {0} instances seen".format(instance_count))
 
if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
      logging.warning(
        "unknown arguments: {0}".format(
            parser.parse_known_args()[1]))
    main(options)

