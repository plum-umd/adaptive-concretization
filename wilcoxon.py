#!/usr/bin/env python

from itertools import combinations
from optparse import OptionParser
import random
import sys

from scipy import stats
from scipy.stats import wilcoxon

from db import PerfDB
import util

smpl_sizes = [8, 12, 16, 20, 24, 28, 32]

def compare(n, data, b, d1, d2):
  d1_space = data[b][d1]["search space"]
  d2_space = data[b][d2]["search space"]
  d1_ttime = data[b][d1]["ttime"]
  d2_ttime = data[b][d2]["ttime"]
  _max_n = min(map(len, [d1_space, d2_space, d1_ttime, d2_ttime]))

  indices = random.sample(range(_max_n), n)
  dist_d1 = [ d1_ttime[idx][1] * d1_space[idx] for idx in indices ]
  dist_d2 = [ d2_ttime[idx][1] * d2_space[idx] for idx in indices ]
  rank_sum, pvalue = wilcoxon(dist_d1, dist_d2)
  return pvalue


def main():
  parser = OptionParser(usage="usage: %prog [options]")
  parser.add_option("--user",
    action="store", dest="user", default="sketchperf",
    help="user name for database")
  parser.add_option("--db",
    action="store", dest="db", default="concretization",
    help="database name")
  parser.add_option("-e", "--eid",
    action="store", dest="eid", type="int", default=11,
    help="experiment id")
  parser.add_option("-b", "--benchmark",
    action="append", dest="benchmarks", default=[],
    help="benchmark(s) of interest")

  (opt, args) = parser.parse_args()

  db = PerfDB(opt.user, opt.db)
  db.drawing = True
  db.detail = True # in particular, needs "search space"
  db.calc_stat(opt.benchmarks, True, opt.eid)
  data = db.raw_data

  merged = util.merge_succ_fail(data)

  for b in merged:
    print "\n=== benchmark: {} ===".format(b)
    ds = merged[b].keys()
    for d1, d2 in combinations(ds, 2):
      for n in smpl_sizes:
        ps = []
        for r in xrange(101):
          p = compare(n, merged, b, d1, d2)
          ps.append(p)
        percentile = util.calc_percentile(ps)
        s_percentile = " | ".join(map(str, percentile))
        print "{} vs. {} w/ {} samples: [{}]".format(d1, d2, n, s_percentile)


if __name__ == "__main__":
  sys.exit(main())

