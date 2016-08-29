# NNJM -- a theano re-implementation of (Vaswani et al. 2013)
# 
# August 2016

import argparse
from collections import Counter
import codecs
import logging
from loss import NCE
from numberizer import numberizer
from numberizer import TARGET_TYPE, SOURCE_TYPE
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

class NNJM:

  # the default noise_distribution is uniform
  def __init__(self, n_gram, vocab_size, word_dim=150, hidden_dim1=512, hidden_dim2=512, noise_sample_size=100, batch_size=1000, noise_dist=[]):
    self.n_gram = n_gram
    self.vocab_size = vocab_size
    self.word_dim = word_dim
    self.hidden_dim1 = hidden_dim1
    self.hidden_dim2 = hidden_dim2
    self.noise_sample_size = noise_sample_size
    self.batch_size = batch_size
    self.noise_dist = noise_dist if noise_dist != [] else np.array([floatX(1. / vocab_size)] * vocab_size, dtype=floatX)
    self.D = theano.shared(
        np.random.uniform(-0.05, 0.05, (word_dim, vocab_size)).astype(floatX),
        name = 'D')
    if hidden_dim1 > 0:
      self.C = theano.shared(
          np.random.uniform(-0.05, 0.05, (hidden_dim1, word_dim * self.ngram)).astype(floatX),
          name = 'C') 
    else:
      pass

    if hidden_dim1 > 0:
      self.M = theano.shared(
          np.random.uniform(-0.05, 0.05, (hidden_dim2, hidden_dim1)).astype(floatX),
          name = 'M')
    else:
      self.M = theano.shared(np.random.uniform(-0.05, 0.05, (hidden_dim2,  word_dim * self.ngram)).astype(floatX), name='M') 

    self.E = theano.shared(
        np.random.uniform(-0.05, 0.05, (vocab_size, hidden_dim2)).astype(floatX),
        name = 'E')

    if hidden_dim1 > 0:
      self.Cb = theano.shared(
          np.array([[-np.log(vocab_size)] * hidden_dim1]).astype(floatX).T,
          name = 'Cb')
    else:
      pass

    self.Mb = theano.shared(
        np.array([[-np.log(vocab_size)] * hidden_dim2]).astype(floatX).T,
        name = 'Mb')

    self.Eb = theano.shared(
        np.array([[-np.log(vocab_size)] * vocab_size]).astype(floatX).T,
        name = 'Eb')

    self.offset = theano.shared(
        np.array(np.arange(0, batch_size * self.vocab_size, self.vocab_size)),
        name = "offset")

    self.__theano_init__()

  def __theano_init__(self):
    X = T.lmatrix('X') # (batch_size, n_gram)
    Y = T.lvector('Y') # (batch_size, )
    # N = T.lmatrix('N') # (batch_size, noise_sample_size)
    # XXX: new NCE loss shares one sample across the whole batch
    N = T.lvector('N')

    #if hidden_dim1 > 0:
    #  CC = T.tile(self.C, (1, self.n_gram)) # (hidden_dim1, word_dim * n_gram)
    #CCb = T.tile(self.Cb, (1, self.batch_size)) # (hidden_dim1, batch_size)
    #else:
    #pass
    #MMb = T.tile(self.Mb, (1, self.batch_size)) # (hidden_dim2, batch_size)
    #EEb = T.tile(self.Eb, (1, self.batch_size)) # (vocab_size, batch_size)
    
    Du = self.D.take(X.T, axis = 1).T # (batch_size, n_gram, word_dim)

    if self.hidden_dim1 > 0:
      h1 = T.nnet.relu(self.C.dot(T.flatten(Du, outdim=2).T) + self.Cb) # (hidden_dim1, batch_size) #TODO: T.flatten look into it...
      h2 = T.nnet.relu(self.M.dot(h1) + Mb) # (hidden_dim2, batch_size)
    else:
      h2 = T.nnet.relu(self.M.dot(T.flatten(Du, outdim=2).T) + self.Mb) # (hidden_dim2, batch_size)

    O = T.exp(self.E.dot(h2) + self.Eb).T # (batch_size, vocab_size)

    predictions = T.argmax(O, axis=1)
    xent = T.sum(T.nnet.categorical_crossentropy(O, Y))

    """
    YY = Y + self.offset # offset indexes used to construct pw and qw
    NN = N + T.tile(self.offset, (self.noise_sample_size, 1)).T # offset indexes used to construct pwb and qwb

    pw = T.take(O, YY) # (batch_size, )
    qw = T.take(self.noise_dist, Y) # (batch_size, )
    pwb = T.take(O, NN) # (batch_size, noise_sample_size)
    qwb = T.take(self.noise_dist, N) # (batch_size, noise_sample_size)
    
    pd1 = pw / (pw + self.noise_sample_size * qw) # (batch_size, )
    pd0 = (self.noise_sample_size * qwb) / (pwb + self.noise_sample_size * qwb) # (batch_size, noise_sample_size)
    """

    # XXX: use new NCE loss introduced in http://www.aclweb.org/anthology/N16-1145
    lossfunc = NCE(self.batch_size, self.vocab_size, self.noise_dist, self.noise_sample_size)
    loss = lossfunc.evaluate(O, Y, N)
    # loss = T.sum(T.log(pd1) + T.sum(T.log(pd0), axis=1)) # scalar

    dD = T.grad(loss, self.D)
    if self.hidden_dim1 > 0:
      dC = T.grad(loss, self.C)
    else:
      pass
    dM = T.grad(loss, self.M)
    dE = T.grad(loss, self.E)
    dCb = T.grad(loss, self.Cb)
    dMb = T.grad(loss, self.Mb)
    dEb = T.grad(loss, self.Eb)
    
    lr = T.scalar('lr', dtype=theano.config.floatX)

    self.pred = theano.function(inputs = [X], outputs = predictions)
    self.xent = theano.function(inputs = [X, Y], outputs = xent)
    self.loss = theano.function(inputs = [X, Y, N], outputs = loss)
    if self.hidden_dim1 > 0:
      self.backprop = theano.function(inputs = [X, Y, N], outputs = [dD, dC, dM, dE, dCb, dMb, dEb])
      self.sgd = theano.function(inputs = [X, Y, N, lr], outputs = [], 
          updates = [
              (self.D, self.D + lr * dD),
              (self.C, self.C + lr * dC),
              (self.M, self.M + lr * dM),
              (self.E, self.E + lr * dE),
              (self.Cb, self.Cb + lr * dCb), 
              (self.Mb, self.Mb + lr * dMb), 
              (self.Eb, self.Eb + lr * dEb), 
              ])
      self.weights = theano.function(inputs = [], outputs = [self.D, self.C, self.M, self.E, self.Cb, self.Mb, self.Eb])
    else:
      self.backprop = theano.function(inputs = [X, Y, N], outputs = [dD, dM, dE, dMb, dEb])
      self.sgd = theano.function(inputs = [X, Y, N, lr], outputs = [], 
          updates = [
              (self.D, self.D + lr * dD),
              (self.M, self.M + lr * dM),
              (self.E, self.E + lr * dE),
              (self.Mb, self.Mb + lr * dMb), 
              (self.Eb, self.Eb + lr * dEb), 
              ])
      self.weights = theano.function(inputs = [], outputs = [self.D,  self.M, self.E,  self.Mb, self.Eb])

  
# ==================== END OF NNJM CLASS DEF ====================

def dump_matrix(m, model_file):
    shape = m.shape
    if len(shape) == 1:
      shape = (shape, 1)

    index = 0
    for i in range(0, shape[0]):
      model_file.write("{0:.6f}".format(m.take(index)))
      for j in range(1, shape[1]):
        index += 1
        model_file.write('\t')
        model_file.write("{0:.6f}".format(m.take(index)))
      model_file.write('\n')

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

    # input_vocab
    #TODO: figure out why this is not dumped in moses' NNJM
    model_file.write("\\input_vocab\n")
    for word in vocab:
      model_file.write(word + "\n")
    model_file.write("\n")
    model_file.write("\\output_vocab\n")
    for word in vocab:
      model_file.write(word + "\n")
    model_file.write("\n")

    if net.hidden_dim1 > 0:
      [D, C, M, E, Cb, Mb, Eb] = net.weights()
    else:
      [D, M, E, Mb, Eb] = net.weights()


    # input_embeddings
    model_file.write("\\input_embeddings\n")
    dump_matrix(D.T, model_file)
    model_file.write("\n")

    if net.hidden_dim1 > 0:
    # hidden_weights 1
      model_file.write("\\hidden_weights 1\n")
      dump_matrix(C.T, model_file)
      model_file.write("\n")

      # hidden_biases 1
      model_file.write("\\hidden_biases 1\n")
      dump_matrix(Cb, model_file)
      model_file.write("\n")
    else:
      pass

    # hidden_weights 2
    model_file.write("\\hidden_weights 2\n")
    dump_matrix(M.T, model_file)
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

def get_left_src(nz, src, a, w):
  lsc =  src[a - w if a - w > 0 else 0: a]
  if len(lsc) < w:
    lsc = [nz.v2i[SOURCE_TYPE, nz.bos]] * (w - len(lsc)) + lsc
  return lsc


def get_right_src(nz, src, a, w):
  rsc = src[a+1: a + 1 + w]
  if len(rsc) < w:
    rsc = rsc + [nz.v2i[SOURCE_TYPE, nz.eos]] * (w - len(rsc))
  return rsc

def get_nearest_src_align(ta2sa, idx):
  assert idx not in ta2sa
  for dist in range(1,100):
    for d in [+1, -1]:
      idx_d_dist = idx + (dist * d)
      if idx_d_dist in ta2sa and len(ta2sa[idx_d_dist]) == 1:
        #if target word is aligned to just one source word
        return ta2sa[idx_d_dist][0]
      elif idx_d_dist in ta2sa and len(ta2sa[idx_d_dist]) > 1:
        #if target word is aligned to many source words, pick the middle alignment
        _s = sorted(ta2sa[idx_d_dist])
        return _s[int(len(_s)/2)]
      else:
        pass

def get_effective_align(align, idx):
  ta2sa = {}
  sa2ta = {}
  for sa,ta in align:
    _s = ta2sa.get(ta, [])
    _s.append(sa)
    ta2sa[ta] = _s
    _t = sa2ta.get(sa, [])
    _t.append(ta)
    sa2ta[sa] = _t
  if idx in ta2sa and len(ta2sa[idx]) == 1:
    #if target word is aligned to just one source word
    return ta2sa[idx][0]
  elif idx in ta2sa and len(ta2sa[idx]) > 1:
    #if target word is aligned to many source words, pick the middle alignment
    _s = sorted(ta2sa[idx])
    return _s[int(len(_s)/2)]
  elif idx not in ta2sa:
    #if the target word is aligned to null
    nearest_sa = get_nearest_trg_align(ta2sa, idx)
    return nearest_sa
  else:
    raise NotImplementedError





def make_training_instances(nz, trnz_align, trnz_target, trnz_source, tc_size=5, sw_size=4):
  input_contexts = []
  output_labels = []
  for trg, src, align in zip(trnz_target, trnz_source, trnz_align):
    for idx in range(1, len(trg)):
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
      input_contexts.append(fullc)
      output_labels.append(trg[idx])
  return input_contexts, output_labels

def main(options):
  # collecting vocab
  logging.info("start collecting vocabulary")
  #indexed_ngrams = []
  #predictions = []
  nz = numberizer(limit = options.vocab_size)
  nz.build_vocab(TARGET_TYPE,options.target_file)
  nz.build_vocab(SOURCE_TYPE,options.source_file)
  trnz_target = nz.numberize_sent(TARGET_TYPE, options.target_file)
  trnz_source = nz.numberize_sent(SOURCE_TYPE, options.source_file)
  trnz_align = read_alignment(options.align_file) 
  input_contexts, output_labels =  make_training_instances(nz, trnz_align, trnz_target, trnz_source, tc_size=options.tc_size, sw_size=options.sw_size)

  target_unigram_counts = np.zeros(len(nz.t2c), dtype=floatX)
  for tw, tw_count in nz.t2c.iteritems():
    t_idx = nz.t2i[tw]
    target_unigram_counts[t_idx] = floatX(tw_count)
  target_unigram_dist  = target_unigram_counts / np.sum(target_unigram_counts)
  logging.info("vocabulary collection finished")

  # training
  if len(vocab) < options.vocab_size:
    logging.warning("The actual vocabulary size of the training corpus {0} ".format(len(vocab)) + 
      "is smaller than the vocab_size option as specified {0}. ".format(options.vocab_size) + 
      "We don't know what will happen to NNJM in that case, but for safety we'll decrease vocab_size as the vocabulary size in the corpus.")
    options.vocab_size = len(vocab)
  logging.info("start training with n-gram size {0}, vocab size {1}, learning rate {2}, "
      .format(options.n_gram, len(vocab), options.learning_rate) + 
      "word dimension {0}, hidden dimension 1 {1}, hidden dimension 2 {2}, noise sample size {3}"
      .format(options.word_dim, options.hidden_dim1, options.hidden_dim2, options.noise_sample_size))
  net = NNJM(options.n_gram, len(vocab), options.word_dim, options.hidden_dim1, options.hidden_dim2,
      options.noise_sample_size, options.batch_size, target_unigram_dist)
  for epoch in range(1, options.max_epoch + 1):
    sgd(input_contexts, output_labels, net, options, epoch, target_unigram_dist)
    if epoch % options.save_interval == 0:
    	dump(net, options.working_dir + "/NNJM.model." + str(epoch), options, vocab)
  logging.info("training finished")

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
          parser.parse_known_args()[1]))
  main(options)

