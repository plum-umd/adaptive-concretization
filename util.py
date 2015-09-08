from functools import partial
import operator as op
import random

import numpy as np

def sanitize_table_name(table_name):
  return table_name.replace('\'', '\'\'')


# benchmark => "'benchmark'"
# 100 => "100"
def quote(x):
  if type(x) in [str, unicode]: return "\'{}\'".format(x)
  else: return str(x)


# "\infty" => "\infty"
# 3.141592654 => "3.14"
def formatter(x, d=0):
  if type(x) is str: return x
  else: return "{0:,.{1}f}".format(x, d)


# initialize dic[k] only if key k is not bound
# i.e., preserve the previous one if mappings for k already exist
def init_k(dic, k):
  if k not in dic: dic[k] = {}


# make a new entry of list type or append the given item
# e.g., {x: [1]}, x, 2 => {x: [1,2]}
#       {x: [1]}, y, 2 => {x: [1], y: [2]}
def mk_or_append(dic, k, v, uniq=False):
  if k in dic: # already bound key
    if not uniq or v not in dic[k]: # uniq => value v not recorded
      dic[k].append(v)
  else: # new occurence of key k
    dic[k] = [v]


# ~ List.split in OCaml
# transform a list of pairs into a pair of lists
# e.g., [ (1, 'a'), (2, 'b'), (3, 'c') ] -> ([1, 2, 3], ['a', 'b', 'c'])
def split(lst):
  if not lst: return ([], [])
  else:
    try:
      a, b = zip(*lst) # zip doesn't handle an empty list
      return (list(a), list(b))
    except ValueError: # [ (1, ), (2, ) ] -> [1, 2]
      return list(zip(*lst)[0])
    except TypeError: # already 1d list
      return lst


# transform a list of pairs into a dict
def to_dict(lst):
  if not lst: return {}
  else: return { k: v for k, v in lst }


# calculate percentiles
# default: quartiles: 0%, 25%, 50%, 75%, 100%
def calc_percentile(lst, ps=[0, 25, 50, 75, 100]):
  _lst = split(lst)
  a = np.array(_lst)
  f = partial(np.percentile, a)
  return map(f, ps)


# calculate semi-interquartile range
def calc_siqr(lst):
  _, q1, q2, q3, _ = calc_percentile(lst)
  siqr = (q3 - q1) / 2
  return q2, siqr


# sort both lists according to the order of the 1st list
def sort_both(l1, l2):
  return zip(*sorted(zip(l1, l2), key=op.itemgetter(0)))


# merge Succeed/Failed data
def merge_succ_fail(data, succ_weight=-1):
  _merged = {}
  for b in data:
    _merged[b] = {}
    for d in data[b]:
      _merged[b][d] = {}
      _merged[b][d]["ttime"] = []
      _max = 0
      if "Succeed" in data[b][d]:
        for t in data[b][d]["Succeed"]:
          _merged[b][d]["ttime"].append( (True, t) )
          if t > _max: _max = t
      if "Failed" in data[b][d]:
        for t in data[b][d]["Failed"]:
          _merged[b][d]["ttime"].append( (False, t) )
          if t > _max: _max = t
      random.shuffle(_merged[b][d])

      # add a default succ case (to avoid zero division)
      if "Succeed" not in data[b][d]:
        if succ_weight > 0:
          _merged[b][d]["ttime"].append( (True, _max * succ_weight) )

      for k in data[b][d]:
        if k in ["Succeed", "Failed"]: continue
        _merged[b][d][k] = data[b][d][k]

  return _merged

