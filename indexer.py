# a general indexer of objects
# but designed with word indexer of a language model in mind
# 
# proudly developed by
# Shuoyang Ding @ Johns Hopkins University
# 
# March, 2016

class indexer:
  def __init__(self):
    self.objects = []
    self.indexes = {}
    self.locked = False

  def getIndex(self, e):
    if e in self.indexes:
      return self.indexes[e]
    else:
      ix = len(self.objects)
      self.indexes[e] = ix
      self.objects.append(e)
      return ix

  def indexOf(self, e):
    if e in self.indexes:
      return self.indexes[e]
    else:
      return -1

  def add(self, e):
    if self.locked == True:
      raise Exception("attempt to add element into a locked indexer")
    if e in self.indexes:
      return False
    else:
      self.indexes[e] = len(self.objects)
      self.objects.append(e)
      return True

  def size(self):
    return len(self.objects)

  def lock(self):
    self.locked = True

  def getObjects(self):
    return self.objects

