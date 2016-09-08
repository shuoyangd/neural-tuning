from numberizer import SOURCE_TYPE, TARGET_TYPE

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

