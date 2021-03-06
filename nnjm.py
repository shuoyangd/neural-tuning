# NNJM -- a theano re-implementation of (Vaswani et al. 2013)
# 
# August 2016

import argparse
from collections import Counter
import codecs
from io import StringIO
import logging
from loss import NCE
from utils.numberizer import numberizer
from utils.numberizer import TARGET_TYPE, SOURCE_TYPE
import numpy as np
import pdb
import cPickle as pickle
import sys
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
parser.add_argument("--target-file", "-tf", dest="target_file", metavar="PATH", help="file with target sentences.")
parser.add_argument("--source-file", "-sf", dest="source_file", metavar="PATH", help="file with source sentences.")
parser.add_argument("--alignment-file", "-af", dest="align_file", metavar="PATH", help="file with word alignments between source-target (giza style).")
parser.add_argument("--model-file", "-mf", dest="model_file", metavar="PATH", help="file with already trained model as a initialization (moses style).")
parser.add_argument("--vocab-file", "-vf", dest="vocab_file", metavar="PATH", help="if you have a pickled dictionary (numberizer), you can use it here.")
parser.add_argument("--working-dir", "-w", dest="working_dir", metavar="PATH", help="Directory used to dump models etc.", required=True)
parser.add_argument("--validation-target-file", "-vtf", dest="val_trg_file", metavar="PATH", help="Validation target corpus used for stopping criteria.")
parser.add_argument("--validation-source-file", "-vsf", dest="val_src_file", metavar="PATH", help="Validation source corpus used for stopping criteria.")
parser.add_argument("--validation-alignment-file", "-vaf", dest="val_align_file", metavar="PATH", help="Validation alignment corpus used for stopping criteria.")
parser.add_argument("--learning-rate", dest="learning_rate", type=float, metavar="FLOAT", help="Learning rate used to update weights (default = 1.0).")
parser.add_argument("--vocab-size", dest="vocab_size", type=int, metavar="INT", help="Vocabulary size on each side of the NNJM  (default = 16000).")
parser.add_argument("--word-dim", dest="word_dim", type=int, metavar="INT", help="Dimension of word embedding (default = 150).")
parser.add_argument("--hidden-dim1", dest="hidden_dim1", type=int, metavar="INT", help="Dimension of hidden layer 1. (default = 750).")
parser.add_argument("--hidden-dim2", dest="hidden_dim2", type=int, metavar="INT", help="Dimension of hidden layer 2. Pass dimension 0 if you only want 1 hidden layer. (default = 0).")
parser.add_argument("--noise-sample-size", "-k", dest="noise_sample_size", type=int, metavar="INT", help="Size of the noise sample per training instance for NCE (default = 100).")
parser.add_argument("--sw-size", dest="sw_size", type=int, metavar="INT", help="Size of the one-side source context (default = 4).")
parser.add_argument("--tc-size", dest="tc_size", type=int, metavar="INT", help="Size of the target context (default = 4).")
parser.add_argument("--max-epoch", dest="max_epoch", type=int, metavar="INT", help="Maximum number of epochs should be performed during training (default = 5).")
parser.add_argument("--save-interval", dest="save_interval", type=int, metavar="INT", help="Saving model only for every several epochs (default = 1).")
parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, metavar="INT", help="Batch size (in sentences) of SGD (default = 1000).")
parser.add_argument("--gradient-check", dest="gradient_check", type=int, metavar="INT", help="The iteration interval for gradient check. Pass 0 if gradient check should not be performed (default = 0).")

parser.set_defaults(
  learning_rate=0.001,
  word_dim=150,
  vocab_size=16000,
  hidden_dim1=0,
  hidden_dim2=750,
  noise_sample_size=100,
  sw_size=4,
  tc_size=4,
  max_epoch=5,
  batch_size=1000,
  save_interval=1)

if theano.config.floatX=='float32':
  floatX = np.float32
else:
  floatX = np.float64

class NNJM:

  def load(self, model_dir, noise_sample_size=100, batch_size=1000, noise_dist=[]):
    self.load_model(model_dir)
    self.noise_sample_size = noise_sample_size
    self.batch_size = batch_size
    self.noise_dist = theano.shared(noise_dist, name='nd') \
        if noise_dist != [] \
        else theano.shared(np.array([floatX(1. / vocab_size)] * vocab_size, dtype=floatX), name = 'nd')

  # the default noise_distribution is uniform
  def __init__(self, num_inputs, vocab_size, target_vocab_size, word_dim=150, hidden_dim1=150, hidden_dim2=750, noise_sample_size=100, batch_size=1000, noise_dist=[]):

    self.num_inputs = num_inputs
    self.vocab_size = vocab_size
    self.target_vocab_size = target_vocab_size
    self.word_dim = word_dim
    self.hidden_dim1 = hidden_dim1
    self.hidden_dim2 = hidden_dim2
    self.noise_sample_size = noise_sample_size
    self.batch_size = batch_size
    self.noise_dist = theano.shared(noise_dist, name='nd') \
        if noise_dist != [] \
        else theano.shared(np.array([floatX(1. / vocab_size)] * vocab_size, dtype=floatX), name = 'nd')
    self.D = theano.shared(
        np.random.uniform(-0.05, 0.05, (word_dim, vocab_size)).astype(floatX),
        name = 'D')
    if hidden_dim1 > 0:
      self.C = theano.shared(
          np.random.uniform(-0.05, 0.05, (hidden_dim1, word_dim * self.num_inputs)).astype(floatX),
          name = 'C') 
    else:
      pass

    if hidden_dim1 > 0:
      self.M = theano.shared(
          np.random.uniform(-0.05, 0.05, (hidden_dim2, hidden_dim1)).astype(floatX),
          name = 'M')
    else:
      self.M = theano.shared(np.random.uniform(-0.05, 0.05, (hidden_dim2,  word_dim * self.num_inputs)).astype(floatX), name='M') 

    self.E = theano.shared(
        np.random.uniform(-0.05, 0.05, (target_vocab_size, hidden_dim2)).astype(floatX),
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
        np.array([[-np.log(target_vocab_size)] * target_vocab_size]).astype(floatX).T,
        name = 'Eb')

    self.__theano_init__()

  def __theano_init__(self):
    self.symX = T.lmatrix('X') # (batch_size, num_inputs)
    self.symY = T.lvector('Y') # (batch_size, )
    # N = T.lmatrix('N') # (batch_size, noise_sample_size)
    # XXX: new NCE loss shares one sample across the whole batch
    self.symN = T.lvector('N')

    if self.hidden_dim1 > 0:
      CCb = T.tile(self.Cb, (1, self.batch_size)) # (hidden_dim1, batch_size)
    else:
      pass

    MMb = T.tile(self.Mb, (1, self.batch_size)) # (hidden_dim2, batch_size)
    EEb = T.tile(self.Eb, (1, self.batch_size)) # (target_vocab_size, batch_size)
    
    Du = self.D.take(self.symX.T, axis = 1).T # (batch_size, num_inputs, word_dim)

    if self.hidden_dim1 > 0:
      h1 = T.nnet.relu(self.C.dot(T.flatten(Du, outdim=2).T) + CCb) # (hidden_dim1, batch_size) #TODO: T.flatten look into it...
      h2 = T.nnet.relu(self.M.dot(h1) + MMb) # (hidden_dim2, batch_size)
    else:
      h2 = T.nnet.relu(self.M.dot(T.flatten(Du, outdim=2).T) + MMb) # (hidden_dim2, batch_size)

    O = T.exp(self.E.dot(h2) + EEb).T # (batch_size, target_vocab_size)

    predictions = T.argmax(O, axis=1)
    xent = T.sum(T.nnet.categorical_crossentropy(O, self.symY))

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
    lossfunc = NCE(self.batch_size, self.target_vocab_size, self.noise_dist, self.noise_sample_size)
    loss = lossfunc.evaluate(O, self.symY, self.symN)
    # loss = T.sum(T.log(pd1) + T.sum(T.log(pd0), axis=1)) # scalar

    self.symdD = T.grad(loss, self.D)
    if self.hidden_dim1 > 0:
      self.symdC = T.grad(loss, self.C)
    else:
      pass
    self.symdM = T.grad(loss, self.M)
    self.symdE = T.grad(loss, self.E)
    if self.hidden_dim1 > 0:
      self.symdCb = T.grad(loss, self.Cb)
    self.symdMb = T.grad(loss, self.Mb)
    self.symdEb = T.grad(loss, self.Eb)
    
    self.symlr = T.scalar('lr', dtype=theano.config.floatX)

    self.pred = theano.function(inputs = [self.symX], outputs = predictions)
    self.xent = theano.function(inputs = [self.symX, self.symY], outputs = xent)
    self.loss = theano.function(inputs = [self.symX, self.symY, self.symN], outputs = loss)
    if self.hidden_dim1 > 0:
      self.backprop = theano.function(inputs = [self.symX, self.symY, self.symN], outputs = [self.symdD, self.symdC, self.symdM, self.symdE, self.symdCb, self.symdMb, self.symdEb])
      self.sgd = theano.function(inputs = [self.symX, self.symY, self.symN, self.symlr], outputs = [], 
          updates = [
              (self.D, self.D + self.symlr * self.symdD),
              (self.C, self.C + self.symlr * self.symdC),
              (self.M, self.M + self.symlr * self.symdM),
              (self.E, self.E + self.symlr * self.symdE),
              (self.Cb, self.Cb + self.symlr * self.symdCb), 
              (self.Mb, self.Mb + self.symlr * self.symdMb), 
              (self.Eb, self.Eb + self.symlr * self.symdEb), 
              ])
      self.weights = theano.function(inputs = [], outputs = [self.D, self.C, self.M, self.E, self.Cb, self.Mb, self.Eb])
      
    else:
      self.backprop = theano.function(inputs = [self.symX, self.symY, self.symN], outputs = [self.symdD, self.symdM, self.symdE, self.symdMb, self.symdEb])
      self.sgd = theano.function(inputs = [self.symX, self.symY, self.symN, self.symlr], outputs = [], 
          updates = [
              (self.D, self.D + self.symlr * self.symdD),
              (self.M, self.M + self.symlr * self.symdM),
              (self.E, self.E + self.symlr * self.symdE),
              (self.Mb, self.Mb + self.symlr * self.symdMb), 
              (self.Eb, self.Eb + self.symlr * self.symdEb), 
              ])
      self.weights = theano.function(inputs = [], outputs = [self.D, self.M, self.E, self.Mb, self.Eb])

  def dump_matrix(self, m, model_file):
      np.savetxt(model_file, m, fmt="%.6f", delimiter='\t')
  
  def dump(self, model_dir):
      model_file = open(model_dir, 'w')
  
      # config
      model_file.write("\\config\n")
      model_file.write("version 1\n")
      model_file.write("ngram_size {0}\n".format(self.num_inputs + 1))
      model_file.write("input_vocab_size {0}\n".format(self.vocab_size))
      model_file.write("output_vocab_size {0}\n".format(self.target_vocab_size))
      model_file.write("input_embedding_dimension {0}\n".format(self.word_dim))
      model_file.write("num_hidden {0}\n".format(self.hidden_dim1))
      model_file.write("output_embedding_dimension {0}\n".format(self.hidden_dim2))
      model_file.write("activation_function rectifier\n\n") # currently only supporting rectifier... 
  
      if self.hidden_dim1 > 0:
        [D, C, M, E, Cb, Mb, Eb] = self.weights()
      else:
        [D, M, E, Mb, Eb] = self.weights()
  
      # input_embeddings
      model_file.write("\\input_embeddings\n")
      self.dump_matrix(np.transpose(D), model_file)
      model_file.write("\n")
  
      # hidden_weights 1
      if self.hidden_dim1 > 0:
        model_file.write("\\hidden_weights 1\n")
        self.dump_matrix(C, model_file)
        model_file.write("\n")
  
        # hidden_biases 1
        model_file.write("\\hidden_biases 1\n")
        self.dump_matrix(Cb, model_file)
        model_file.write("\n")
      else:
        # hidden_weights 2
        model_file.write("\\hidden_weights 1\n")
        self.dump_matrix(M, model_file)
        model_file.write("\n")
    
        # hidden_biases 2
        model_file.write("\\hidden_biases 1\n")
        self.dump_matrix(Mb, model_file)
        model_file.write("\n")
  
      # Made compliant to Moses-accepted format
      # Note hidden_dim1 in the options is defined differently as in model file
      if self.hidden_dim1 > 0: 
        # hidden_weights 2
        model_file.write("\\hidden_weights 2\n")
        self.dump_matrix(M, model_file)
        model_file.write("\n")
    
        # hidden_biases 2
        model_file.write("\\hidden_biases 2\n")
        self.dump_matrix(Mb, model_file)
        model_file.write("\n")
      else:
        model_file.write("\\hidden_weights 2\n")
        model_file.write("0.5\n")
        model_file.write("\n")
        
        model_file.write("\\hidden_biases 2\n")
        model_file.write("0.5\n")
        model_file.write("\n")
  
      # output_weights
      model_file.write("\\output_weights\n")
      self.dump_matrix(E, model_file)
      model_file.write("\n")
  
      # output_biases
      model_file.write("\\output_biases\n")
      self.dump_matrix(Eb, model_file)
      model_file.write("\n")
  
      model_file.write("\\end")
      model_file.close()

  def load_matrix(self, model_file):
    line = model_file.readline()
    mstr = ""
    while line.strip() != "":
      mstr += line
      line = model_file.readline()
    logging.info("read all lines for this matrix")
    mstrio = StringIO(unicode(mstr))
    return np.loadtxt(mstrio)

  def load_model(self, model_dir):
    model_file = open(model_dir)

    logging.info("loading model...")
    line = model_file.readline()
    while line.strip() != "\\end":
      if line.strip() == "\\config":
        line = model_file.readline()
        while line.strip() != "":
          pair = line.strip().split(' ')
          if pair[0] == "ngram_size":
            self.num_inputs = int(pair[1]) - 1
          if pair[0] == "input_vocab_size":
            self.vocab_size = int(pair[1])
          if pair[0] == "output_vocab_size":
            self.target_vocab_size = int(pair[1])
          if pair[0] == "input_embedding_dimension":
            self.word_dim = int(pair[1])
          if pair[0] == "num_hidden":
            self.hidden_dim1 = int(pair[1])
          if pair[0] == "output_embedding_dimension":
            self.hidden_dim2 = int(pair[1])
          logging.info("config loaded")
          line = model_file.readline()
      if line.strip() == "\\input_embeddings":
        logging.info("input loading...")
        self.D = np.transpose(self.load_matrix(model_file))
        logging.info("input loaded")
      if line.strip() == "\\hidden_weights 1":
        logging.info("hidden_W1 loading...")
        self.C = self.load_matrix(model_file)
        logging.info("hidden_W1 loaded")
      if line.strip() == "\\hidden_biases 1":
        logging.info("hidden_b1 loading...")
        self.Cb = self.load_matrix(model_file)
        logging.info("hidden_b1 loaded")
      if line.strip() == "\\hidden_weights 2":
	logging.info("hidden_W2 loading...")
        self.M = self.load_matrix(model_file)
	logging.info("hidden_W2 loaded")
      if line.strip() == "\\hidden_biases 2":
	logging.info("hidden_b2 loading...")
        self.Mb = self.load_matrix(model_file)
	logging.info("hidden_b2 loaded")
      if line.strip() == "\\output_weights":
	logging.info("output_W loading...")
        self.E = self.load_matrix(model_file)
	logging.info("output_W loaded")
      if line.strip() == "\\output_biases":
	logging.info("output_b loading...")
        self.Eb = self.load_matrix(model_file)
	logging.info("output_b loaded")
      line = model_file.readline()

# ==================== END OF NNJM CLASS DEF ====================

def shuffle(indexed_ngrams, predictions):
  logging.info("shuffling data... ")
  arr = np.arange(len(indexed_ngrams))
  np.random.shuffle(arr)
  indexed_ngrams_shuffled = indexed_ngrams[arr, :]
  predictions_shuffled = predictions[arr]
  return (indexed_ngrams_shuffled, predictions_shuffled)

def sgd(indexed_ngrams, predictions, net, options, epoch, noise_dist):
  logging.info("epoch {0} started".format(epoch))  
  instance_count = 0
  batch_count = 0
  # for performance issue, if the remaining data is smaller than batch_size, we just discard them
  for start in range(0, len(indexed_ngrams), options.batch_size):
    if len(indexed_ngrams) - start >= options.batch_size:
      X = indexed_ngrams[start: start + options.batch_size]
      Y = predictions[start: start + options.batch_size]
      N = np.array(rand.distint(noise_dist, (options.noise_sample_size,)), dtype='int64') # (noise_sample_size, )
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

def validate(indexed_ngrams, predictions, net, options, epoch, noise_dist):
  xent = 0.0
  loss = 0.0
  for start in range(0, len(indexed_ngrams), options.batch_size):
    if len(indexed_ngrams) - start >= options.batch_size:
      X = indexed_ngrams[start: start + options.batch_size]
      Y = predictions[start: start + options.batch_size]
      N = np.array(rand.distint(noise_dist, (options.noise_sample_size,)), dtype='int64') # (noise_sample_size, )
      xent += net.xent(X, Y)
      loss += net.loss(X, Y, N)
  logging.info("validation upon completing epoch {0}: cross entropy {1}, NCE loss {2}".format(epoch, xent, loss))

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
    nearest_sa = get_nearest_src_align(ta2sa, idx)
    return nearest_sa
  else:
    raise NotImplementedError

def make_training_instances(nz, trnz_align, trnz_target, trnz_source, tc_size=4, sw_size=4):
  input_contexts = []
  output_labels = []
  linen = 0
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
    linen += 1
  return np.array(input_contexts), np.array(output_labels)

def main(options):
  options.n_gram = options.sw_size * 2 + options.tc_size + 2
  # collecting vocab
  logging.info("start collecting vocabulary")
  #indexed_ngrams = []
  #predictions = []
  if not options.vocab_file:
    nz = numberizer(limit = options.vocab_size)
    nz.build_vocab(TARGET_TYPE,options.target_file)
    nz.build_vocab(SOURCE_TYPE,options.source_file)
  else:
    nz = pickle.load(open(options.vocab_file))
  
  trnz_target = nz.numberize_sent(TARGET_TYPE, options.target_file)
  trnz_source = nz.numberize_sent(SOURCE_TYPE, options.source_file)
  trnz_align = read_alignment(options.align_file) 
  if not options.vocab_file:
    pickle.dump(nz, open(options.working_dir + "/numberizer.pickle", 'wb'))
  input_contexts, output_labels =  make_training_instances(nz, trnz_align, trnz_target, trnz_source, tc_size=options.tc_size, sw_size=options.sw_size)
  if options.val_trg_file and options.val_src_file and options.val_align_file:
    vanz_target = nz.numberize_sent(TARGET_TYPE, options.val_trg_file)
    vanz_source = nz.numberize_sent(SOURCE_TYPE, options.val_src_file)
    vanz_align = read_alignment(options.val_align_file)
    val_input_contexts, val_output_labels =  make_training_instances(nz, vanz_align, vanz_target, vanz_source, tc_size=options.tc_size, sw_size=options.sw_size)
  elif options.val_trg_file or options.val_src_file or options.val_align_file:
    logging.fatal("You have to supply all three validation files (source, target, alignment) to trigger validation.")
    sys.exit(1)

  target_unigram_counts = np.zeros(len(nz.t2c), dtype=floatX)
  for tw, tw_count in nz.t2c.iteritems():
    t_idx = nz.t2i[tw]
    target_unigram_counts[t_idx] = floatX(tw_count)
  target_unigram_dist  = target_unigram_counts / np.sum(target_unigram_counts)
  logging.info("vocabulary collection finished")

  # training
  if len(nz.v2i) < 2 * options.vocab_size:
    logging.warning("The actual vocabulary size of the training corpus {0} ".format(len(nz.v2i)) + 
      "is smaller than the vocab_size option as specified {0}. ".format(options.vocab_size) + 
      "We don't know what will happen to NNJM in that case, but for safety we'll decrease vocab_size as the vocabulary size in the corpus.")
  options.vocab_size = len(nz.v2i)
  options.target_vocab_size = len(nz.t2i)

  logging.info("start training with n-gram size {0}, vocab size {1}, learning rate {2}, "
      .format(options.n_gram, len(nz.v2i), options.learning_rate) + 
      "word dimension {0}, hidden dimension 1 {1}, hidden dimension 2 {2}, noise sample size {3}"
      .format(options.word_dim, options.hidden_dim1, options.hidden_dim2, options.noise_sample_size))
  net = NNJM(options.n_gram - 1, len(nz.v2i), len(nz.t2i), options.word_dim, options.hidden_dim1, options.hidden_dim2,
      options.noise_sample_size, options.batch_size, target_unigram_dist)
  if not options.model_file == None:
    net.load(options.model_file, options.noise_sample_size, options.batch_size, target_unigram_dist)
  nz.save_vocab_in_moses_format(options.working_dir + "/vocab")
  for epoch in range(1, options.max_epoch + 1):
    (input_contexts_shuffled, output_labels_shuffled) = shuffle(input_contexts, output_labels)
    sgd(input_contexts_shuffled, output_labels_shuffled, net, options, epoch, target_unigram_dist)
    if options.val_trg_file and options.val_src_file and options.val_align_file:
      validate(val_input_contexts, val_output_labels, net, options, epoch, target_unigram_dist)
    if epoch % options.save_interval == 0:
      net.dump(options.working_dir + "/NNJM.model." + str(epoch))
  logging.info("training finished")

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
          parser.parse_known_args()[1]))
  main(options)

