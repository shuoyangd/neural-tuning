# nplm -- a theano re-implementation of (Vaswani et al. 2013)
# To improve numerical stability, we used Adadelta (Zeiler 2012) as optimization method.
# 
# proudly developed by
# Shuoyang Ding @ Johns Hopkins University
# 
# April, 2016

import argparse
from indexer import indexer
import logging
from numberizer import numberizer
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
parser.add_argument("--training-file", "-t", dest="training_file", metavar="PATH", help="File used as training corpus.", required=True)
parser.add_argument("--working-dir", "-w", dest="working_dir", metavar="PATH", help="Directory used to dump models etc.", required=True)
# parser.add_argument("--validation-file", dest="validation_file", metavar="PATH", help="Validation corpus used for stopping criteria.")
parser.add_argument("--decay-rate", dest="decay_rate", type=float, metavar="FLOAT", help="Decay rate as required by Adadelta (default = 0.95).")
parser.add_argument("--epsilon", "-e", dest="epsilon", type=float, metavar="FLOAT", help="Constant epsilon as required by Adadelta (default = 1e-6).")
parser.add_argument("--vocab-size", dest="vocab_size", type=int, metavar="INT", help="Vocabulary size of the language model (default = 500000).")
parser.add_argument("--word-dim", dest="word_dim", type=int, metavar="INT", help="Dimension of word embedding (default = 150).")
parser.add_argument("--hidden-dim1", dest="hidden_dim1", type=int, metavar="INT", help="Dimension of hidden layer 1 (default = 150).")
parser.add_argument("--hidden-dim2", dest="hidden_dim2", type=int, metavar="INT", help="Dimension of hidden layer 2 (default = 750).")
parser.add_argument("--noise-sample-size", "-k", dest="noise_sample_size", type=int, metavar="INT", help="Size of the noise sample per training instance for NCE (default = 100).")
parser.add_argument("--n-gram", "-n", dest="n_gram", type=int, metavar="INT", help="Size of the N-gram (default = 5).")
parser.add_argument("--max-epoch", dest="max_epoch", type=int, metavar="INT", help="Maximum number of epochs should be performed during training (default = 5).")
parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, metavar="INT", help="Batch size (in sentences) of SGD (default = 1000).")
parser.add_argument("--save-interval", dest="save_interval", type=int, metavar="INT", help="Saving model only for every several epochs (default = 1).")

parser.set_defaults(
  decay_rate=0.95,
  epsilon=1e-6,
  vocab_size=500000,
  word_dim=150,
  hidden_dim1=150,
  hidden_dim2=750,
  noise_sample_size=100,
  n_gram=5,
  max_epoch=5,
  batch_size=1000,
  save_interval=1)

if theano.config.floatX=='float32':
  floatX = np.float32
else:
  floatX = np.float64

class nplm:

  # the default noise_distribution is uniform
  def __init__(self, n_gram, vocab_size, word_dim=150, hidden_dim1=150, hidden_dim2=750, noise_sample_size=100, batch_size=1000, decay_rate=0.95, epsilon=1e-6, noise_dist=[]):
    self.n_gram = n_gram
    self.vocab_size = vocab_size
    self.word_dim = word_dim
    self.hidden_dim1 = hidden_dim1
    self.hidden_dim2 = hidden_dim2
    self.noise_sample_size = noise_sample_size
    self.batch_size = batch_size
    self.decay_rate = decay_rate
    self.epsilon = epsilon
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
    self.Cb = theano.shared(
        np.array([[-np.log(vocab_size)] * hidden_dim1]).astype(floatX).T,
        name = 'Cb')
    self.Mb = theano.shared(
        np.array([[-np.log(vocab_size)] * hidden_dim2]).astype(floatX).T,
        name = 'Mb')
    self.Eb = theano.shared(
        np.array([[-np.log(vocab_size)] * vocab_size]).astype(floatX).T,
        name = 'Eb')
    self.offset = theano.shared(
        np.array(np.arange(0, batch_size * self.vocab_size, self.vocab_size)),
        name = "offset")

    # Adadelta parameters kept for each variable
    self.D_eg2 = theano.shared(np.zeros((word_dim, vocab_size)).astype(floatX), name='D_eg2')
    self.D_edx2 = theano.shared(np.zeros((word_dim, vocab_size)).astype(floatX), name='D_edx2')
    self.C_eg2 = theano.shared(np.zeros((hidden_dim1, word_dim)).astype(floatX), name='C_eg2')
    self.C_edx2 = theano.shared(np.zeros((hidden_dim1, word_dim)).astype(floatX), name='C_edx2')
    self.M_eg2 = theano.shared(np.zeros((hidden_dim2, hidden_dim1)).astype(floatX), name='M_eg2')
    self.M_edx2 = theano.shared(np.zeros((hidden_dim2, hidden_dim1)).astype(floatX), name='M_edx2')
    self.E_eg2 = theano.shared(np.zeros((vocab_size, hidden_dim2)).astype(floatX), name='E_eg2')
    self.E_edx2 = theano.shared(np.zeros((vocab_size, hidden_dim2)).astype(floatX), name='E_edx2')
    self.Cb_eg2 = theano.shared(np.array([[0.0] * hidden_dim1]).astype(floatX).T, name='Cb_eg2')
    self.Cb_edx2 = theano.shared(np.array([[0.0] * hidden_dim1]).astype(floatX).T, name='Cb_edx2')
    self.Mb_eg2 = theano.shared(np.array([[0.0] * hidden_dim2]).astype(floatX).T, name='Mb_eg2')
    self.Mb_edx2 = theano.shared(np.array([[0.0] * hidden_dim2]).astype(floatX).T, name='Mb_edx2')
    self.Eb_eg2 = theano.shared(np.array([[0.0] * vocab_size]).astype(floatX).T, name='Eb_eg2')
    self.Eb_edx2 = theano.shared(np.array([[0.0] * vocab_size]).astype(floatX).T, name='Eb_edx2')

    self.__theano_init__()

  def __theano_init__(self):
    X = T.lmatrix('X') # (batch_size, n_gram)
    Y = T.lvector('Y') # (batch_size, )
    N = T.lmatrix('N') # (batch_size, noise_sample_size)
    CC = T.tile(self.C, (1, self.n_gram)) # (hidden_dim1, word_dim * n_gram)
    CCb = T.tile(self.Cb, (1, self.batch_size)) # (hidden_dim1, batch_size)
    MMb = T.tile(self.Mb, (1, self.batch_size)) # (hidden_dim2, batch_size)
    EEb = T.tile(self.Eb, (1, self.batch_size)) # (vocab_size, batch_size)
    
    Du = self.D.take(X.T, axis = 1).T # (batch_size, n_gram, word_dim)
    h1 = T.nnet.relu(CC.dot(T.flatten(Du, outdim=2).T) + CCb) # (hidden_dim1, batch_size)
    h2 = T.nnet.relu(self.M.dot(h1) + MMb) # (hidden_dim2, batch_size)
    O = T.exp(self.E.dot(h2) + EEb).T # (batch_size, vocab_size)

    predictions = T.argmax(O, axis=1)
    xent = T.sum(T.nnet.categorical_crossentropy(O, Y))

    YY = Y + self.offset # offset indexes used to construct pw and qw
    NN = N + T.tile(self.offset, (self.noise_sample_size, 1)).T # offset indexes used to construct pwb and qwb

    pw = T.take(O, YY) # (batch_size, )
    qw = T.take(self.noise_dist, Y) # (batch_size, )
    pwb = T.take(O, NN) # (batch_size, noise_sample_size)
    qwb = T.take(self.noise_dist, N) # (batch_size, noise_sample_size)

    pd1 = pw / (pw + self.noise_sample_size * qw) # (batch_size, )
    pd0 = (self.noise_sample_size * qwb) / (pwb + self.noise_sample_size * qwb) # (batch_size, noise_sample_size)
    
    loss = T.sum(T.log(pd1) + T.sum(T.log(pd0), axis=1)) # scalar
    gD = T.grad(loss, self.D)
    gC = T.grad(loss, self.C)
    gM = T.grad(loss, self.M)
    gE = T.grad(loss, self.E)
    gCb = T.grad(loss, self.Cb)
    gMb = T.grad(loss, self.Mb)
    gEb = T.grad(loss, self.Eb)
    
    D_eg2 = self.decay_rate * self.D_eg2 + (1 - self.decay_rate) * gD * gD
    C_eg2 = self.decay_rate * self.C_eg2 + (1 - self.decay_rate) * gC * gC
    M_eg2 = self.decay_rate * self.M_eg2 + (1 - self.decay_rate) * gM * gM
    E_eg2 = self.decay_rate * self.E_eg2 + (1 - self.decay_rate) * gE * gE
    Cb_eg2 = self.decay_rate * self.Cb_eg2 + (1 - self.decay_rate) * gCb * gCb
    Mb_eg2 = self.decay_rate * self.Mb_eg2 + (1 - self.decay_rate) * gMb * gMb
    Eb_eg2 = self.decay_rate * self.Eb_eg2 + (1 - self.decay_rate) * gEb * gEb

    dD = self.rms(self.D_edx2) / self.rms(D_eg2) * gD
    dC = self.rms(self.C_edx2) / self.rms(C_eg2) * gC
    dM = self.rms(self.M_edx2) / self.rms(M_eg2) * gM
    dE = self.rms(self.E_edx2) / self.rms(E_eg2) * gE
    dCb = self.rms(self.Cb_edx2) / self.rms(Cb_eg2) * gCb
    dMb = self.rms(self.Mb_edx2) / self.rms(Mb_eg2) * gMb
    dEb = self.rms(self.Eb_edx2) / self.rms(Eb_eg2) * gEb

    self.pred = theano.function(inputs = [X], outputs = predictions)
    self.xent = theano.function(inputs = [X, Y], outputs = xent)
    self.loss = theano.function(inputs = [X, Y, N], outputs = loss)
    self.backprop = theano.function(inputs = [X, Y, N], outputs = [dD, dC, dM, dE, dCb, dMb, dEb])
    # pdb.set_trace()
    self.sgd = theano.function(inputs = [X, Y, N], outputs = [], 
        updates = [
            (self.D, self.D + dD),
            (self.C, self.C + dC),
            (self.M, self.M + dM),
            (self.E, self.E + dE),
            (self.Cb, self.Cb + dCb),
            (self.Mb, self.Mb + dMb),
            (self.Eb, self.Eb + dEb),
            # eg2
	    (self.D_eg2, D_eg2),
	    (self.C_eg2, C_eg2),
	    (self.M_eg2, M_eg2),
	    (self.E_eg2, E_eg2),
	    (self.Cb_eg2, Cb_eg2),
	    (self.Mb_eg2, Mb_eg2),
	    (self.Eb_eg2, Eb_eg2),
            # edx2
	    (self.D_edx2, self.decay_rate * self.D_edx2 + (1 - self.decay_rate) * dD * dD),
	    (self.C_edx2, self.decay_rate * self.C_edx2 + (1 - self.decay_rate) * dC * dC),
	    (self.M_edx2, self.decay_rate * self.M_edx2 + (1 - self.decay_rate) * dM * dM),
	    (self.E_edx2, self.decay_rate * self.E_edx2 + (1 - self.decay_rate) * dE * dE),
	    (self.Cb_edx2, self.decay_rate * self.Cb_edx2 + (1 - self.decay_rate) * dCb * dCb), 
	    (self.Mb_edx2, self.decay_rate * self.Mb_edx2 + (1 - self.decay_rate) * dMb * dMb), 
	    (self.Eb_edx2, self.decay_rate * self.Eb_edx2 + (1 - self.decay_rate) * dEb * dEb)
            ])
    self.weights = theano.function(inputs = [], outputs = [self.D, self.C, self.M, self.E, self.Cb, self.Mb, self.Eb])

  def rms(self, e2):
    return T.sqrt(e2 + self.epsilon)

# ==================== END OF NPLM CLASS DEF ====================

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
    model_file.write("num_hidden {0}\n".format(options.hidden_dim1))
    model_file.write("output_embedding_dimension {0}\n".format(options.hidden_dim2))
    model_file.write("activation_function rectifier\n\n") # currently only supporting rectifier... 

    # input_vocab
    model_file.write("\\input_vocab\n")
    for word in vocab:
      model_file.write(word + "\n")
    model_file.write("\n")
    model_file.write("\\output_vocab\n")
    for word in vocab:
      model_file.write(word + "\n")
    model_file.write("\n")

    [D, C, M, E, Cb, Mb, Eb] = net.weights()

    # input_embeddings
    model_file.write("\\input_embeddings\n")
    dump_matrix(D.T, model_file)
    model_file.write("\n")

    # hidden_weights 1
    model_file.write("\\hidden_weights 1\n")
    dump_matrix(C.T, model_file)
    model_file.write("\n")

    # hidden_biases 1
    model_file.write("\\hidden_biases 1\n")
    dump_matrix(Cb, model_file)
    model_file.write("\n")

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
      N = np.array(rand.distint(noise_dist, (options.batch_size, options.noise_sample_size)), dtype='int64') # (batch_size, noise_sample_size)
      net.sgd(X, Y, N)
      instance_count += options.batch_size
      batch_count += 1
      if batch_count % 1 == 0:
        logging.info("{0} instances seen".format(instance_count))
  # N = np.array(rand.distint(noise_dist, (len(indexed_ngrams), options.noise_sample_size)))
  # total_loss = net.loss(indexed_ngrams, predictions, N)
  # logging.info("epoch {0} finished with NCE loss {1}".format(epoch, total_loss))
  logging.info("epoch {0} finished".format(epoch))

def main(options):
  # collecting vocab
  logging.info("start collecting vocabulary")
  indexed_ngrams = []
  predictions = []
  nz = numberizer(limit = options.vocab_size, unk = UNK, bos = BOS, eos = EOS)
  (trnz, vocab, unigram_count) = nz.numberize(options.training_file)
  bos_index = vocab.index(BOS)
  eos_index = vocab.index(EOS)
  for numberized_line in trnz:
    # think of a sentence with only 1 word w0 and we are extracting trigrams (n_gram = 3):
    # the numerized version would be "<s> w0 </s>".
    # after the sentence is augmented with 1 extra "<s>" at the beginning (now has length 4), 
    # we want to extract 1 trigram: [<s>, <s>, w0] (note that we don't want [<s>, w0, </s>])
    indexed_sentence = [bos_index] * (options.n_gram - 2)
    indexed_sentence.extend(numberized_line)
    for start in range(len(indexed_sentence) - options.n_gram):
      indexed_ngrams.append(indexed_sentence[start: start + options.n_gram])
      if start + options.n_gram < len(indexed_sentence):
        predictions.append(indexed_sentence[start + options.n_gram])
  del trnz

  # build quick vocab indexer
  v2i = {}
  for i in range(len(vocab)):
    v2i[vocab[i]] = i

  total_unigram_count = floatX(sum(unigram_count.values()))
  unigram_dist = [floatX(0.0)] * len(unigram_count)
  # pdb.set_trace()
  for key in unigram_count.keys():
    unigram_dist[v2i[key]] = floatX(unigram_count[key] / total_unigram_count)
  del unigram_count
  unigram_dist = np.array(unigram_dist, dtype=floatX)
  logging.info("vocabulary collection finished")

  # training
  if len(vocab) < options.vocab_size:
    logging.warning("The actual vocabulary size of the training corpus {0} ".format(len(vocab)) + 
      "is smaller than the vocab_size option as specified {0}. ".format(options.vocab_size) + 
      "We don't know what will happen to nplm in that case, but for safety we'll decrease vocab_size as the vocabulary size in the corpus.")
    options.vocab_size = len(vocab)
  logging.info("start training with n-gram size {0}, vocab size {1}, decay_rate {2}, epsilon {3}, "
      .format(options.n_gram, len(vocab), options.decay_rate, options.epsilon) + 
      "word dimension {0}, hidden dimension 1 {1}, hidden dimension 2 {2}, noise sample size {3}"
      .format(options.word_dim, options.hidden_dim1, options.hidden_dim2, options.noise_sample_size))
  net = nplm(options.n_gram, len(vocab), options.word_dim, options.hidden_dim1, options.hidden_dim2,
      options.noise_sample_size, options.batch_size, options.decay_rate, options.epsilon, unigram_dist)
  for epoch in range(1, options.max_epoch + 1):
    sgd(indexed_ngrams, predictions, net, options, epoch, unigram_dist)
    if epoch % options.save_interval == 0:
        logging.info("dumping models")
    	dump(net, options.working_dir + "/nplm.model." + str(epoch), options, vocab)
  logging.info("training finished")

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
          parser.parse_known_args()[1]))
  main(options)

