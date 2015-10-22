#!/usr/bin/env python

from itertools import combinations, repeat, chain
from optparse import OptionParser
import os
import random
import sys

from scipy import stats
from scipy.stats import wilcoxon, ranksums

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from db import PerfDB
import util

smpl_sizes = [8, 16, 24, 32]

def compare(n, data, b, d1, d2):
  d1_space = data[b][d1]["search space"] # [ s1_1, s1_2, s1_3, ... ]
  d2_space = data[b][d2]["search space"] # [ s2_1, s2_2, s2_3, ... ]
  d1_ttime = data[b][d1]["ttime"] # [ (_, t1_1), (_, t1_2), (_, t1_3), ... ]
  d2_ttime = data[b][d2]["ttime"] # [ (_, t2_1), (_, t2_2), (_, t2_3), ... ]
  # space/ttime may have different length of data
  # so, pick the smallest one to avoid out-of-index error
  _max_n = min(map(len, [d1_space, d2_space, d1_ttime, d2_ttime]))

  # pick n random numbers between 0 to the size of the smallest data set
  pop = range(_max_n)
  try:
    indices = random.sample(pop, n)
  except ValueError: # sample larger than population
    indices = list(chain.from_iterable(repeat(pop, n / _max_n))) + random.sample(pop, n % _max_n)
  except TypeError: # single value
    indices = repeat(_max_n, n)

  # expected running time = t/p, where p = 1/search space
  dist_d1 = [ d1_ttime[idx][1] * d1_space[idx] for idx in indices ]
  dist_d2 = [ d2_ttime[idx][1] * d2_space[idx] for idx in indices ]
  rank_sum, pvalue = wilcoxon(dist_d1, dist_d2)
  return pvalue


def draw_bubble_chart(out_dir, b, xs, ys, ps):
  cm = plt.cm.get_cmap('jet')

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_xscale("log", nonposx="clip")
  ax.set_yscale("log", nonposx="clip")

  # bubble area: pi * r'^2 (r' : normalized w.r.t. 0.05)
  areas = [ np.pi * ((p / 0.05) ** 2) for p in ps ]
  sc = ax.scatter(xs, ys, s=areas, c=ps, cmap=cm, linewidth=0, alpha=0.5)
  fig.colorbar(sc)

  png = os.path.join(out_dir, "wilcoxon_{}.png".format(b))
  print "drawing", png
  plt.savefig(png)
  plt.close()


def main():
  parser = OptionParser(usage="usage: %prog [options]")
  parser.add_option("-c", "--cmd",
    action="store", dest="cmd",
    type="choice", choices=["degree", "perf"],
    default=None, help="command to run")
  parser.add_option("--user",
    action="store", dest="user", default="sketchperf",
    help="user name for database")
  parser.add_option("--db",
    action="store", dest="db", default="concretization",
    help="database name")
  parser.add_option("-e", "--eid",
    action="append", dest="eids", type="int", default=[],
    help="experiment id")
  parser.add_option("-b", "--benchmark",
    action="append", dest="benchmarks", default=[],
    help="benchmark(s) of interest")
  parser.add_option("-d", "--dir",
    action="store", dest="data_dir", default="data",
    help="output folder")
  parser.add_option("-v", "--verbose",
    action="store_true", dest="verbose", default=False,
    help="verbosely print out data to be drawn")

  (opt, args) = parser.parse_args()

  if not opt.cmd:
    parser.error("nothing to do")

  db = PerfDB(opt.user, opt.db)
  db.drawing = True
  db.detail_space = True
  if not opt.eids: opt.eids = [11]

  # degree comparisons
  if opt.cmd == "degree":
    db.calc_stat(opt.benchmarks, True, opt.eids)
    merged = util.merge_succ_fail(db.raw_data)

    for b in merged:
      if opt.verbose:
        print "\n=== benchmark: {} ===".format(b)

      # collecting (d1, d2, p) to draw bubble charts
      xs = []
      ys = []
      ps = []

      ds = sorted(merged[b].keys())
      for d1, d2 in combinations(ds, 2):
        for n in smpl_sizes:
          _ps = []
          for r in xrange(301):
            _p = compare(n, merged, b, d1, d2)
            _ps.append(_p)
          percentile = util.calc_percentile(_ps)
          if opt.verbose:
            s_percentile = " | ".join(map(str, percentile))
            print "{} vs. {} w/ {} samples: [{}]".format(d1, d2, n, s_percentile)
          xs.append(d1)
          ys.append(d2)
          # pick the median, programmatically
          ps.append(percentile[len(percentile)/2])

      draw_bubble_chart(opt.data_dir, b, xs, ys, ps)

  # performance comparisons
  elif opt.cmd == "perf":
    if len(opt.eids) <= 1:
      parser.error("requires at least two data sets")

    data = {}
    for eid in opt.eids:
      print "collecting data at EID={}".format(eid)
      db.calc_stat(opt.benchmarks, False, [eid])
      data[eid] = db.raw_data
      db.reset_raw_data()

    eids = sorted(data.keys())
    for e1, e2 in combinations(eids, 2):
      print "\n=== EID={} vs. EID={} ===".format(e1, e2)
      bs = set(data[e1].keys() + data[e2].keys())
      x = []
      y = []
      for b in bs:
        # consider a benchmark that appears at both data sets
        if (b not in data[e1]) or (b not in data[e2]): continue
        _x = util.find_all(data[e1][b], "TTIME")
        _y = util.find_all(data[e2][b], "TTIME")
        s, p = ranksums(_x, _y)
        tab = '\t' * (3 - len(b)/8)
        print "{}{} ranksum stat: {},\tp-value: {}".format(b, tab, util.formatter(s, 4), util.formatter(p, 4))
        x.append(_x) #x.append(sorted(_x))
        y.append(_y) #y.append(sorted(_y))
        #x.append(util.calc_percentile(_x, [50]))
        #y.append(util.calc_percentile(_y, [50]))
      fx, fy = map(util.flatten, [x, y])
      s, p = ranksums(fx, fy)
      print "overall\t\t\t ranksum stat: {},\tp-value: {}".format(util.formatter(s, 4), util.formatter(p, 4))


if __name__ == "__main__":
  sys.exit(main())

