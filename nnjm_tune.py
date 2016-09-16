# NNJM -- a theano re-implementation of (Vaswani et al. 2013)
# 
# August 2016

import argparse
from collections import Counter
import codecs
import logging
from loss import NCE
from nnjm import NNJM
from utils.numberizer import numberizer
from utils.numberizer import TARGET_TYPE, SOURCE_TYPE
import numpy as np
import pdb
import theano
import theano.tensor as T
import rand

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

# constants
UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"

parser = argparse.ArgumentParser()
parser.add_argument("--target-file", "-t", dest="target_file", metavar="PATH", help="file with target sentences.", required=True)
parser.add_argument("--source-file", "-s", dest="source_file", metavar="PATH", help="file with source sentences.", required=True)
parser.add_argument("--alignment-file", "-a", dest="align_file", metavar="PATH", help="file with word alignments between source-target (giza style).", required=True)
parser.add_argument("--working-dir", "-w", dest="working_dir", metavar="PATH", help="Directory used to dump models etc.", required=True)
parser.add_argument("--validation-file", dest="validation_file", metavar="PATH", help="Validation corpus used for stopping criteria.")
parser.add_argument("--learning-rate", dest="learning_rate", type=float, metavar="FLOAT", help="Learning rate used to update weights (default = 1.0).")
parser.add_argument("--vocab-size", dest="vocab_size", type=int, metavar="INT", help="Vocabulary size of the language model (default = 500000).")
parser.add_argument("--word-dim", dest="word_dim", type=int, metavar="INT", help="Dimension of word embedding (default = 150).")
parser.add_argument("--hidden-dim1", dest="hidden_dim1", type=int, metavar="INT", help="Dimension of hidden layer 1 (default = 150).")
parser.add_argument("--hidden-dim2", dest="hidden_dim2", type=int, metavar="INT", help="Dimension of hidden layer 2 (default = 750).")
parser.add_argument("--noise-sample-size", "-k", dest="noise_sample_size", type=int, metavar="INT", help="Size of the noise sample per training instance for NCE (default = 100).")
parser.add_argument("--n-gram", "-n", dest="n_gram", type=int, metavar="INT", help="Size of the N-gram (default = 5).")
parser.add_argument("--max-epoch", dest="max_epoch", type=int, metavar="INT", help="Maximum number of epochs should be performed during training (default = 5).")
parser.add_argument("--save-interval", dest="save_interval", type=int, metavar="INT", help="Saving model only for every several epochs (default = 1).")
parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, metavar="INT", help="Batch size (in sentences) of SGD (default = 1000).")
parser.add_argument("--gradient-check", dest="gradient_check", type=int, metavar="INT", help="The iteration interval for gradient check. Pass 0 if gradient check should not be performed (default = 0).")

parser.set_defaults(
  learning_rate=0.001,
  word_dim=150,
  vocab_size=500000,
  hidden_dim1=512,
  hidden_dim2=512,
  noise_sample_size=100,
  sw_size=4,
  tc_size=5,
  n_gram=14,
  max_epoch=5,
  batch_size=128,
  save_interval=1)

if theano.config.floatX=='float32':
  floatX = np.float32
else:
  floatX = np.float64
  
# ==================== END OF NNJM CLASS DEF ====================

def dump_matrix(m, model_file):
    np.savetxt(model_file, m, fmt="%.6f", delimiter='\t')

def dump_vocab(vocab_dir, vocab):
    src_vocab_file = open(vocab_dir + ".source", 'w')
    trg_vocab_file = open(vocab_dir + ".target", 'w') 
    for word in vocab:
      if word[0] == TARGET_TYPE: 
        src_vocab_file.write((word[1] + "\n").encode('utf-8'))
      elif word[0] == SOURCE_TYPE:
        trg_vocab_file.write((word[1] + "\n").encode('utf-8'))
    src_vocab_file.write('\n')
    trg_vocab_file.write('\n')
    src_vocab_file.close()
    trg_vocab_file.close()

def dump(net, model_dir, options, vocab):
    model_file = open(model_dir, 'w')

    # config
    model_file.write("\\config\n")
    model_file.write("version 1\n")
    model_file.write("ngram_size {0}\n".format(options.n_gram))
    model_file.write("input_vocab_size {0}\n".format(options.vocab_size))
    model_file.write("output_vocab_size {0}\n".format(options.vocab_size))
    model_file.write("input_embedding_dimension {0}\n".format(options.word_dim))
    model_file.write("num_hidden {0}\n".format(options.hidden_dim1)) #TODO: does not match NNJM in moses!!
    model_file.write("output_embedding_dimension {0}\n".format(options.hidden_dim2))
    model_file.write("activation_function rectifier\n\n") # currently only supporting rectifier... 

    if net.hidden_dim1 > 0:
      [D, C, M, E, Cb, Mb, Eb] = net.weights()
    else:
      [D, M, E, Mb, Eb] = net.weights()

    # input_embeddings
    model_file.write("\\input_embeddings\n")
    dump_matrix(np.transpose(D), model_file)
    model_file.write("\n")

    if net.hidden_dim1 > 0:
    # hidden_weights 1
      model_file.write("\\hidden_weights 1\n")
      dump_matrix(np.transpose(C), model_file)
      model_file.write("\n")

      # hidden_biases 1
      model_file.write("\\hidden_biases 1\n")
      dump_matrix(Cb, model_file)
      model_file.write("\n")
    else:
      pass

    # hidden_weights 2
    model_file.write("\\hidden_weights 2\n")
    dump_matrix(np.transpose(M), model_file)
    model_file.write("\n")

    # hidden_biases 2
    model_file.write("\\hidden_biases 2\n")
    dump_matrix(Mb, model_file)
    model_file.write("\n")

    # output_weights
    model_file.write("\\output_weights\n")
    dump_matrix(E, model_file)
    model_file.write("\n")

    # output_biases
    model_file.write("\\output_biases\n")
    dump_matrix(Eb, model_file)
    model_file.write("\n")

    model_file.write("\\end")
    model_file.close()

def sgd(indexed_ngrams, predictions, net, options, epoch, noise_dist):
  logging.info("epoch {0} started".format(epoch))  
  instance_count = 0
  batch_count = 0
  # for performance issue, if the remaining data is smaller than batch_size, we just discard them
  for start in range(0, len(indexed_ngrams), options.batch_size):
    if len(indexed_ngrams) - start >= options.batch_size:
      X = indexed_ngrams[start: start + options.batch_size]
      Y = predictions[start: start + options.batch_size]
      N = np.array(rand.distint(noise_dist, (options.noise_sample_size,)), dtype='int64') # (batch_size, noise_sample_size)
      # N = np.array(rand.distint(noise_dist, (options.batch_size, options.noise_sample_size)), dtype='int64') # (batch_size, noise_sample_size)
      net.sgd(X, Y, N, floatX(options.learning_rate))
    instance_count += options.batch_size
    batch_count += 1
    if batch_count % 1 == 0:
      logging.info("{0} instances seen".format(instance_count))
  # N = np.array(rand.distint(noise_dist, (len(indexed_ngrams), options.noise_sample_size)))
  # total_loss = net.loss(indexed_ngrams, predictions, N)
  # logging.info("epoch {0} finished with NCE loss {1}".format(epoch, total_loss))
  logging.info("epoch {0} finished".format(epoch))

def read_alignment(align_file):
  n_align = []
  with open(align_file) as f:
    for line in f:
      n = [(int(t.split('-')[0]), int(t.split('-')[1])) for t in line.strip().split()]
      n_align.append(n)
  return n_align


def get_training_tuple(idx, trg, src, align, tc_size, sw_size): 
  tc = [] # contains target context
  sc = [] # contains source context
  tc = trg[idx - tc_size if idx - tc_size > 0 else 0 :idx]
  if len(tc) < tc_size:
    tc_pad = [nz.v2i[TARGET_TYPE,nz.bos]] * (tc_size - len(tc))
    tc = tc_pad + tc
  assert len(tc) == tc_size
  h_a = get_effective_align(align, idx)
  if h_a < len(src):
    pass
  else:
    pdb.set_trace()
  sc = get_left_src(nz, src, h_a,sw_size) 
  sc += [src[h_a]] 
  sc += get_right_src(nz, src, h_a, sw_size)
  fullc = sc + tc
  assert len(fullc) == tc_size + 1 + (2 * sw_size)
  return fullc, trg[idx]

def make_tuning_instances(nz, n_best_alignments, n_best_targets, n_best_sources,n=200, tc_size=5, sw_size=4):
  positive_input_contexts = []
  positive_output_labels = []
  negative_input_contexts = []
  negative_output_labels = []
  for n_idx, (trg, src, align) in enumerate(zip(n_best_targets, n_best_sources, n_best_alignments)):
    if n_idx % 200 == 0: # convert this to a range if we want top k and bottom k in the n-best 
      for idx in range(1, len(trg)):
        fullc, trg_idx = get_training_tuple(idx, trg, src, align, tc_size, sw_size)
        positive_input_contexts.append(fullc)
        positive_output_labels.append(trg[idx])
    elif n_idx % 200 == 199:
      for idx in range(1, len(trg)):
          fullc, trg_idx = get_training_tuple(idx, trg, src, align, tc_size, sw_size)
          negative_input_contexts.append(fullc)
          negative_output_labels.append(trg_idx)
    else:
        pass
  #TODO: make positive and negative example same size (instance size) by creating dummy input,output instances using unk or <s>
  #TODO: also do the same padding such that we have a nice number of examples integer of examples...
  return positive_input_contexts, positive_output_labels, negative_input_contexts, negative_output_labels


def main(options):
  # collecting vocab
  logging.info("start collecting vocabulary")
  #indexed_ngrams = []
  #predictions = []
  nz = numberizer()
  nz.load('path/to/numberized/corpus')
  #nz = numberizer(limit = options.vocab_size)
  #nz.build_vocab(TARGET_TYPE,options.target_file)
  #nz.build_vocab(SOURCE_TYPE,options.source_file)
  trnz_target = nz.numberize_sent(TARGET_TYPE, options.target_file)
  trnz_source = nz.numberize_sent(SOURCE_TYPE, options.source_file)
  trnz_align = read_alignment(options.align_file) 
  pos_contexts, pos_outputs, neg_contexts, neg_outputs = make_tuning_instances(nz,trnz_align, trnz_target, trnz_source) 
  #TODO: proceed from this point on...
  #input_contexts, output_labels =  make_training_instances(nz, trnz_align, trnz_target, trnz_source, tc_size=options.tc_size, sw_size=options.sw_size)

  target_unigram_counts = np.zeros(len(nz.t2c), dtype=floatX)
  for tw, tw_count in nz.t2c.iteritems():
    t_idx = nz.t2i[tw]
    target_unigram_counts[t_idx] = floatX(tw_count)
  target_unigram_dist  = target_unigram_counts / np.sum(target_unigram_counts)
  logging.info("vocabulary collection finished")

  # training
  if len(nz.v2i) < options.vocab_size:
    logging.warning("The actual vocabulary size of the training corpus {0} ".format(len(nz.v2i)) + 
      "is smaller than the vocab_size option as specified {0}. ".format(options.vocab_size) + 
      "We don't know what will happen to NNJM in that case, but for safety we'll decrease vocab_size as the vocabulary size in the corpus.")
    options.vocab_size = len(nz.v2i)
  logging.info("start training with n-gram size {0}, vocab size {1}, learning rate {2}, "
      .format(options.n_gram, len(nz.v2i), options.learning_rate) + 
      "word dimension {0}, hidden dimension 1 {1}, hidden dimension 2 {2}, noise sample size {3}"
      .format(options.word_dim, options.hidden_dim1, options.hidden_dim2, options.noise_sample_size))
  net = NNJM(options.n_gram, len(nz.v2i), options.word_dim, options.hidden_dim1, options.hidden_dim2,
      options.noise_sample_size, options.batch_size, target_unigram_dist)
  dump_vocab(options.working_dir + "/vocab", nz.v2i.keys())
  for epoch in range(1, options.max_epoch + 1):
    sgd(input_contexts, output_labels, net, options, epoch, target_unigram_dist)
    if epoch % options.save_interval == 0:
    	dump(net, options.working_dir + "/NNJM.model." + str(epoch), options, nz.v2i.keys())
  logging.info("training finished")

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
          parser.parse_known_args()[1]))
  main(options)

