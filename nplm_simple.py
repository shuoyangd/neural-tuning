# nplm -- a theano re-implementation of (Vaswani et al. 2013)
# 
# proudly developed by
# Shuoyang Ding @ Johns Hopkins University
# 
# March, 2016

import argparse
from collections import Counter
from indexer import indexer
import logging
import numpy as np
import pdb
import theano
import theano.tensor as T
import rand

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--training-file", "-t", dest="training_file", metavar="PATH", help="File used as training corpus.", required=True)
parser.add_argument("--working-dir", "-w", dest="working_dir", metavar="PATH", help="Directory used to dump models etc.", required=True)
parser.add_argument("--validation-file", dest="validation_file", metavar="PATH", help="Validation corpus used for stopping criteria.")
parser.add_argument("--learning-rate", dest="learning_rate", type=float, metavar="FLOAT", help="Learning rate used to update weights (default = 1.0).")
parser.add_argument("--word-dim", dest="word_dim", type=int, metavar="INT", help="Dimension of word embedding (default = 150).")
parser.add_argument("--hidden-dim1", dest="hidden_dim1", type=int, metavar="INT", help="Dimension of hidden layer 1 (default = 150).")
parser.add_argument("--hidden-dim2", dest="hidden_dim2", type=int, metavar="INT", help="Dimension of hidden layer 2 (default = 750).")
parser.add_argument("--noise-sample-size", "-k", dest="noise_sample_size", type=int, metavar="INT", help="Size of the noise sample per training instance for NCE (default = 100).")
parser.add_argument("--n-gram", "-n", dest="n_gram", type=int, metavar="INT", help="Size of the N-gram (default = 5).")
parser.add_argument("--max-epoch", dest="max_epoch", type=int, metavar="INT", help="Maximum number of epochs should be performed during training (default = 5).")
parser.add_argument("--save-interval", dest="save_interval", type=int, metavar="INT", help="The epoch interval for saving models. Pass 0 if wish to save only once at the end of each epoch (default = 0).")
parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, metavar="INT", help="Batch size (in sentences) of SGD (default = 1000).")
parser.add_argument("--gradient-check", dest="gradient_check", type=int, metavar="INT", help="The iteration interval for gradient check. Pass 0 if gradient check should not be performed (default = 0).")

parser.set_defaults(
  learning_rate=1.0,
  word_dim=150,
  hidden_dim1=150,
  hidden_dim2=750,
  noise_sample_size=100,
  n_gram=5,
  max_epoch=5,
  saving_interval=0,
  batch_size=1000,
  gradient_check=0)

if theano.config.floatX=='float32':
  floatX = np.float32
else:
  floatX = np.float64

class nplm:

  # the default noise_distribution is uniform
  def __init__(self, n_gram, vocab_size, word_dim=150, hidden_dim1=150, hidden_dim2=750, noise_sample_size=100, batch_size=1000, noise_dist=[]):
    self.n_gram = n_gram
    self.vocab_size = vocab_size
    self.word_dim = word_dim
    self.hidden_dim1 = hidden_dim1
    self.hidden_dim2 = hidden_dim2
    self.noise_sample_size = noise_sample_size
    self.batch_size = batch_size
    self.noise_dist = noise_dist if noise_dist != [] else np.array([floatX(1. / vocab_size)] * vocab_size, dtype=floatX)
    self.D = theano.shared(
        np.random.uniform(-0.01, 0.01, (word_dim, vocab_size)).astype(floatX),
        name = 'D')
    self.C = theano.shared(
        np.random.uniform(-0.01, 0.01, (hidden_dim1, word_dim)).astype(floatX),
        name = 'C')
    self.M = theano.shared(
        np.random.uniform(-0.01, 0.01, (hidden_dim2, hidden_dim1)).astype(floatX),
        name = 'M')
    self.E = theano.shared(
        np.random.uniform(-0.01, 0.01, (vocab_size, hidden_dim2)).astype(floatX),
        name = 'E')
    self.b = theano.shared(
        np.array([[-np.log(vocab_size)] * vocab_size]).astype(floatX).T,
        name = 'b')

    self.__theano_init__()

  def __theano_init__(self):
    X = T.imatrix('X') # (batch_size, n_gram)
    Y = T.ivector('Y') # (batch_size, )
    N = T.imatrix('N') # (batch_size, noise_sample_size)
    CC = T.tile(self.C, (1, self.n_gram)) # (hidden_dim1, word_dim * n_gram)

    # x is integer vector representing the n-gram -- each element is an index of a word
    def fprop_step(x, D, CC, M, E, b, n_gram, word_dim):
      Du = D.take(x, axis=1)
      h1 = T.nnet.relu(CC.dot(T.reshape(Du, (n_gram * word_dim, 1))))
      h2 = T.nnet.relu(M.dot(h1))
      o = T.exp(E.dot(h2) + b).T[0] # r for raw distribution, a.k.a unnormalized 
      return o / T.sum(o)

    (O, _) = theano.scan(fn = fprop_step,
      sequences = X,
      outputs_info = None,
      non_sequences = [self.D, CC, self.M, self.E, self.b, self.n_gram, self.word_dim],
      strict=True) # (batch_size, vocab_size)
    predictions = T.argmax(O, axis=1)
    loss = T.sum(T.nnet.categorical_crossentropy(O, Y))

    """
    offset = T.arange(0, X.shape[0] * self.vocab_size, self.vocab_size) # (batch_size, )
    YY = Y + offset # offset indexes used to construct pw and qw
    NN = N + T.tile(offset, (self.noise_sample_size, 1)).T # offset indexes used to construct pwb and qwb

    # pdb.set_trace()
    pw = T.take(O, YY) # (batch_size, )
    qw = T.take(self.noise_dist, Y) # (batch_size, )
    pwb = T.take(O, NN) # (batch_size, noise_sample_size)
    qwb = T.take(self.noise_dist, N) # (batch_size, noise_sample_size)
    
    pd1 = pw / (pw + self.noise_sample_size * qw) # (batch_size, )
    pd0 = (self.noise_sample_size * qwb) / (pwb + self.noise_sample_size * qwb) # (batch_size, noise_sample_size)

    loss = T.sum(T.log(pd1) + T.sum(T.log(pd0), axis=1)) # scalar
    """

    dD = T.grad(loss, self.D)
    dC = T.grad(loss, self.C)
    dM = T.grad(loss, self.M)
    dE = T.grad(loss, self.E)
    db = T.grad(loss, self.b)
    
    lr = T.scalar('lr', dtype=theano.config.floatX)

    # self.rawo = theano.function(inputs = [X], outputs = O)
    self.pred = theano.function(inputs = [X], outputs = predictions)
    # self.xent = theano.function(inputs = [X, Y], outputs = xent)
    # self.loss = theano.function(inputs = [X, Y, N], outputs = loss)
    self.loss = theano.function(inputs = [X, Y], outputs = loss)
    # self.backprop = theano.function(inputs = [X, Y, N], outputs = [dD, dC, dM, dE, db])
    self.backprop = theano.function(inputs = [X, Y], outputs = [dD, dC, dM, dE, db])
    # self.sgd = theano.function(inputs = [X, Y, N, lr], outputs = [], 
    self.sgd = theano.function(inputs = [X, Y, lr], outputs = [], 
        updates = [
            (self.D, self.D - lr * dD),
            (self.C, self.C - lr * dC),
            (self.M, self.M - lr * dM),
            (self.E, self.E - lr * dE),
            (self.b, self.b - lr * db)
            ])

def sgd(indexed_ngrams, predictions, net, options, epoch, noise_dist):
  logging.info("epoch {0} started".format(epoch))  
  instance_count = 0
  batch_count = 0
  for start in range(0, len(indexed_ngrams), options.batch_size):
    X = indexed_ngrams[start: min(start + options.batch_size, len(indexed_ngrams))]
    Y = predictions[start: min(start + options.batch_size, len(indexed_ngrams))]
    # N = np.array(rand.distint(noise_dist, (min(options.batch_size, len(indexed_ngrams) - start), options.noise_sample_size)), dtype='int8') # (batch_size, noise_sample_size)
    # net.sgd(X, Y, N, floatX(options.learning_rate))
    net.sgd(X, Y, floatX(options.learning_rate))
    # pdb.set_trace()
    instance_count += min(options.batch_size, len(indexed_ngrams) - start)
    batch_count += 1
    if batch_count % 1 == 0:
      logging.info("{0} instances seen".format(instance_count))
    # if options.gradient_check != 0 and batch_count % options.gradient_check == 0:
    #   net.gradient_check(x, y)
    # if options.save_interval != 0 and batch_count % options.save_interval:
      # supposed to save model here
    #   pass
  # N = np.array(rand.distint(noise_dist, (len(indexed_ngrams), options.noise_sample_size)), dtype='int8')
  # total_loss = net.loss(indexed_ngrams, predictions, N)
  total_loss = net.loss(indexed_ngrams, predictions)
  logging.info("epoch {0} finished with NCE loss {1}".format(epoch, total_loss))
  logging.info("epoch {0} finished".format(epoch))

def main(options):
  # collecting vocab
  logging.info("start collecting vocabulary")
  training_corpus = open(options.training_file)
  indexed_ngrams = []
  predictions = []
  vocab = indexer()
  vocab.add("</s>") # end = 0
  vocab.add("<s>") # start = 1
  unigram_count = Counter()
  sent_count = 0
  for sentence in training_corpus:
    tokens = ["<s>"] * (options.n_gram - 1)
    tokens.extend(sentence.strip().split(' '))
    indexed_sentence = []
    for token in tokens:
      ix = vocab.getIndex(token)
      indexed_sentence.append(ix)
      if token != "<s>":
        count = unigram_count.get(ix, 0)
        unigram_count[ix] = count + 1
    # think of a sentence with length 1 and we are extracting bigrams:
    # after the sentence is augmented with extra "<s>" at the beginning (now has length 2), 
    # we want to extract 1 bigrams: [<s>, w0] (note that we don't want [w0, </s>])
    # that's why we add 1 here.
    for start in range(len(indexed_sentence) - options.n_gram + 1):
      indexed_ngrams.append(indexed_sentence[start: start + options.n_gram])
      if start + options.n_gram < len(indexed_sentence):
        predictions.append(indexed_sentence[start + options.n_gram])
      else:
        eix = vocab.indexOf("</s>")
        predictions.append(eix)
        count = unigram_count.get(eix, 0)
        unigram_count[eix] = count + 1
    sent_count += 1
  unigram_count[vocab.getIndex("<s>")] = sent_count
  training_corpus.close() 

  total_unigram_count = floatX(sum(unigram_count.values()))
  unigram_dist = [floatX(0.0)] * len(unigram_count)
  for key in unigram_count.keys():
    unigram_dist[key] = floatX(unigram_count[key] / total_unigram_count)
  unigram_count.clear() # save some memory... 
  unigram_dist = np.array(unigram_dist, dtype=floatX)
  logging.info("vocabulary collection finished")

  # training
  logging.info("start training with n-gram size {0}, vocab size {1}, learning rate {2}, "
      .format(options.n_gram, vocab.size(), options.learning_rate) + 
      "word dimension {0}, hidden dimension 1 {1}, hidden dimension 2 {2}, noise sample size {3}"
      .format(options.word_dim, options.hidden_dim1, options.hidden_dim2, options.noise_sample_size))
  net = nplm(options.n_gram, vocab.size(), options.word_dim, options.hidden_dim1, options.hidden_dim2,
      options.noise_sample_size, options.batch_size, unigram_dist)
  for epoch in range(options.max_epoch):
    sgd(indexed_ngrams, predictions, net, options, epoch, unigram_dist)
  logging.info("training finished")

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
          parser.parse_known_args()[1]))
  main(options)

