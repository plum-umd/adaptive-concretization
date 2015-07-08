from functools import partial
import operator as op

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


def calc_percentile(lst, ps=[0, 25, 50, 75, 100]):
  _lst = split(lst)
  a = np.array(_lst)
  f = partial(np.percentile, a)
  return map(f, ps)


def calc_siqr(lst):
  _, q1, q2, q3, _ = calc_percentile(lst)
  siqr = (q3 - q1) / 2
  return q2, siqr


# sort both lists according to the order of the 1st list
def sort_both(l1, l2):
  return zip(*sorted(zip(l1, l2), key=op.itemgetter(0)))

