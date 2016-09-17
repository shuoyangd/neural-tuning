# numberizer -- numberize a tokenized text corpora and optionally truncate the vocabulary
# (substitute the rest with a special symbol)
# designed with language model application in mind
# 
# proundly developed by
# Shuoyang Ding @ Johns Hopkins University
# 
# April, 2016

from collections import Counter
import codecs
import logging
import operator
import cPickle as pickle
import sys

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
global TARGET_TYPE, SOURCE_TYPE
TARGET_TYPE='t'
SOURCE_TYPE='s'
class numberizer:

  # vocabulary limit = 0 means no vocabulary truncating will be performed
  # setting either start or end to "None" will stop the numberizer
  #   from adding start and ending symbol to the sentence
  # 
  # the reason to set these symbols is to avoid undesirable collisions with tokens
  #   that are actually part of the corpus
  def __init__(self, limit=0, unk="<unk>", bos="<s>", eos="</s>"):
    self.limit = limit
    self.unk = unk
    self.bos = bos
    self.eos = eos
    self.dont_augment_bos_eos = not bos or not eos
    self.v2i = {} # contains both target and source words and maps to an id key (word_type, word) --> value id
    self.t2i = {}
    self.t2c = {}
    self.s2c = {}
    self.s2i = {}

  def save(self, save_file):
    save_file = open(save_file, 'wb')
    pickle.dump(self, save_file)
    save_file.flush()
    save_file.close()
  
  def load(self, load_file):
    load_file = open(load_file, 'rb')
    self = picke.load(load_file)
    load_file.close()

  def save_vocab_in_moses_format(self, save_vocab_file):
      raise NotImplementedError()

  def build_vocab(self,vocab_type, text_file):
    # self.v2i[vocab_type, self.bos] = self.v2i.get((vocab_type, self.bos), len(self.v2i))
    # self.v2i[vocab_type, self.eos] = self.v2i.get((vocab_type, self.eos), len(self.v2i))
    # self.v2i[vocab_type, self.unk] = self.v2i.get((vocab_type, self.unk), len(self.v2i))
    w2c = {} #temporarily holds words to count...
    with codecs.open(text_file, 'r', 'utf8') as f:
      for line in f:
        # augment bos and eos
        if vocab_type == TARGET_TYPE:
            w2c[self.bos] = w2c.get(self.bos, 0) + 1
            w2c[self.eos] = w2c.get(self.eos, 0) + 1
        else: 
            w2c[self.bos] = w2c.get(self.bos, 0) + 1 #self.s2i.get(self.bos, len(self.s2i))
            w2c[self.eos] = w2c.get(self.eos, 0) + 1 #self.s2i.get(self.eos, len(self.s2i))
        # collect "real" vocab
        for word in line.split():
          #self.v2i[vocab_type, word] = self.v2i.get((vocab_type, word), len(self.v2i))
          if vocab_type == TARGET_TYPE:
            w2c[word] = w2c.get(word, 0) + 1
            #self.t2i[word] = self.t2i.get(word, len(self.t2i))
          else:
            w2c[word] = w2c.get(word, 0) + 1
            #self.s2i[word] = self.s2i.get(word, len(self.s2i))

    if vocab_type == TARGET_TYPE:
        count_sorted = sorted(w2c.items(), key=operator.itemgetter(1), reverse=True) 
        for (w,c)  in count_sorted[:self.limit]:
            self.t2c[w] = c
            self.v2i[vocab_type, w] = self.v2i.get((vocab_type, w), len(self.v2i))
            self.t2i[w] = self.v2i[vocab_type, w]
        self.v2i[vocab_type, self.unk] = self.v2i.get((vocab_type, self.unk), len(self.v2i))
        self.t2i[self.unk] = self.v2i[vocab_type, self.unk]
        self.t2c[self.unk] = self.v2i[vocab_type, self.unk]
    else:
        count_sorted = sorted(w2c.items(), key=operator.itemgetter(1), reverse=True) 
        for (w,c)  in count_sorted[:self.limit]: #TODO: should we limit source vocab also?
            self.s2c[w] = c
            self.v2i[vocab_type, w] = self.v2i.get((vocab_type, w), len(self.v2i))
            self.s2i[w] = self.v2i[vocab_type, w]
        self.v2i[vocab_type, self.unk] = self.v2i.get((vocab_type, self.unk), len(self.v2i))
        self.s2i[self.unk] = self.v2i[vocab_type, self.unk]
        self.s2c[self.unk] = self.v2i[vocab_type, self.unk]

  def numberize_sent(self,vocab_type, text_file):
    n_sent = []
    with codecs.open(text_file, 'r', 'utf8') as f:
      for line in f:
        # n = [self.v2i[vocab_type,w] for w in line.split()]
        n = [self.v2i.get((vocab_type, w), self.v2i[vocab_type, self.unk]) for w in line.split()]
        n = [self.v2i[vocab_type, self.bos]] + n + [self.v2i[vocab_type, self.eos]]
        n_sent.append(n)
    return n_sent

  # the three returned values are:
  # + numberized corpus
  # + a list of vocabulary: you can use it as an indexer -- 
  #     it's guaranteed to provide the same index as the numberized corpus
  # + a counter of the raw tokens (not numberized but truncated) 
  def numberize(self, text_dir, numberized_dir = None):
    # first scan: collect
    text_file = open(text_dir)
    cnt = Counter()
    if self.limit != 0:
      vocab = [self.unk]
    else:
      vocab = []
    linen = 1
    logging.info("Starting first scan of the training corpus.")
    for line in text_file:
      if linen % 100000 == 0:
        logging.info("{0} lines scanned.".format(linen))
      if not self.dont_augment_bos_eos:
        cnt[self.bos] += 1
        cnt[self.eos] += 1
      tokens = line.strip().split(' ')
      for token in tokens:
        cnt[token] += 1
      linen += 1
    if self.limit != 0:
      pairs = cnt.most_common(self.limit - 1) # leave a space for <unk>
      vocab.extend([pair[0] for pair in pairs])
    else:
      vocab.extend(list(cnt.elements()))
    text_file.close()
    logging.info("First scan of the training corpus finished.")

    # build fast indexer
    vocab_indexer = {}
    for i in range(len(vocab)):
      vocab_indexer[vocab[i]] = i

    # remove stop words from counter and add their counts to unk
    if self.limit != 0:
      cnt[self.unk] = 0 # should have unk in counter anyway to keep length consistent
      for key in cnt.keys():
        if not key in vocab_indexer:
          oov_count = cnt[key]
          del cnt[key]
          cnt[self.unk] += oov_count

    # second scan: numberize and truncate
    text_file = open(text_dir)
    numberized = []
    unk_index = vocab_indexer[self.unk]
    bos_index = vocab_indexer[self.bos]
    eos_index = vocab_indexer[self.eos]
    linen = 1
    logging.info("Starting second scan of the training corpus.")
    for line in text_file:
      if linen % 100000 == 0:
        logging.info("{0} lines scanned.".format(linen))
      numberized_line = []
      if not self.dont_augment_bos_eos:
        numberized_line.append(bos_index)
      tokens = line.strip().split(' ')
      for token in tokens:
        if self.limit == 0: # if vocab truncating is not imposed, don't bother
          numberized_line.append(vocab_indexer[token])
        elif token in vocab_indexer: # in-vocabulary
          numberized_line.append(vocab_indexer[token])
        else: # OOV
          numberized_line.append(unk_index)
      if not self.dont_augment_bos_eos:
        numberized_line.append(eos_index)
      numberized.append(numberized_line)
      linen += 1
    text_file.close()
    del vocab_indexer
    logging.info("Second scan of the training corpus finished.")

    # dump
    if numberized_dir:
      numberized_file = open(numberized_dir)
      pickle.dump(numberized, numberized_file)
      numberized_file.close()

    return (numberized, vocab, cnt)

