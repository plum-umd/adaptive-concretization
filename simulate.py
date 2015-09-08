#!/usr/bin/env python

from itertools import repeat
from functools import partial
import random
from optparse import OptionParser
import sys

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon

from db import PerfDB
import util

verbose = False

# default-ish degrees
degrees = [16, 64, 128, 512, 1024, 4096]


def sampling(data, b, d, n=1):
  res = {}
  for k in data[b][d]:
    _bdk = data[b][d][k]
    try:
      res[k] = random.sample(_bdk, n)
    except ValueError: # sample larger than population
      k_len = len(_bdk)
      res[k] = list(repeat(_bdk, k_len / n)) + random.sample(_bdk, k_len % n)
    except TypeError: # single value
      res[k] = _bdk
  #print "data: {}".format(res)
  return res


def sim_with_degree(sampler, n_cpu, s_name, d, n_runs = 0):
  ttime = 0
  found = False
  while not found:
    runs = sampler(d, n_cpu)
    _n_runs = 0
    _ttime = 0
    for s, t in runs["ttime"]:
      _n_runs = _n_runs + 1
      _ttime = _ttime + t
      found = found | s
      if found:
        #print "{} found a solution within {} trials".format(s_name, n_runs+_n_runs)
        _ttime = t
        break

    n_runs = n_runs + _n_runs
    ttime = ttime + _ttime

  return ttime


def test_runs(sampler, n_cpu, s_name, ds):
  ttime = 0
  model = {}
  found = False
  n_runs = 0
  for d in ds:
    model[d] = {}
    model[d]["runs"] = []
    t_runs = sampler(d, n_cpu/2)

    _n_runs = 0
    _ttime = 0
    for s, t in t_runs["ttime"]:
      model[d]["runs"].append(t)
      _n_runs = _n_runs + 1
      _ttime = _ttime + t
      found = found | s
      if found:
        #print "{} found a solution while {} test runs".format(s_name, n_runs+_n_runs)
        _ttime = t
        break

    n_runs = n_runs + _n_runs
    model[d]["ttime"] = _ttime
    model[d]["p"] = t_runs["p"]

    ttime = ttime + _ttime
    if found: break

  return ttime, found, n_runs, model


def strategy_fixed(d, sampler, n_cpu):
  return sim_with_degree(sampler, n_cpu, "strategy_fixed_()".format(d), d)


def strategy_random(sampler, n_cpu):
  # pick a degree randomly
  d = random.choice(degrees)
  #if verbose: print "strategy_random, pick degree: {}".format(d)

  return sim_with_degree(sampler, n_cpu, "strategy_random", d)


def strategy_time(f, msg, sampler, n_cpu):
  ttime, found, n_runs, model = test_runs(sampler, n_cpu, msg, degrees)

  # resampling with likelihood degree
  if not found:
    est = []
    ds = []
    for d in model:
      est.append(model[d]["ttime"])
      ds.append(d)
    idx = est.index(f(est))
    d = ds[idx]
    if verbose: print "{}, pick degree: {}".format(msg, d)

    ttime = ttime + sim_with_degree(sampler, n_cpu, msg, d, n_runs)

  return ttime


def strategy_wilcoxon(sampler, n_cpu):
  pivots = [0, len(degrees)-1]
  ds = [ degrees[pivot] for pivot in pivots ]
  ttime, found, n_runs, model = test_runs(sampler, n_cpu, "strategy_wilcoxon", ds)

  d = None
  fixed = False
  while not found and not fixed and pivots[0] < pivots[1]:
    fixed = True
    d1, d2 = ds
    # TODO: replace empirical p w/ 1/seaerch space
    p1, p2 = [ model[d]["p"] for d in ds ]
    if p1 == 0: # move left pivot to right
      pivots[0] = pivots[0] + 1
      fixed = False
    elif p2 == 0: # move right pivot to left
      pivots[1] = pivots[1] - 1
      fixed = False
    else:
      dist_d1 = [ t1 / p1 for t1 in model[d1]["runs"] ]
      dist_d2 = [ t2 / p2 for t2 in model[d2]["runs"] ]
      rank_sum, pvalue = wilcoxon(dist_d1, dist_d2)
      if pvalue < 0.2: # the median diff. is significatly different
        if np.mean(dist_d1) < np.mean(dist_d2):
          # left one is better, so move right pivot to left
          pivots[1] = pivots[1] - 1
        else: pivots[0] = pivots[0] + 1
        fixed = False
      else: # i.e., can't differentiate two degrees, break the loop
        if np.mean(dist_d1) < np.mean(dist_d2): d = d1
        else: d = d2

    if not fixed: # try another degree
      ds = [ degrees[pivot] for pivot in pivots ]
      _ttime, found, n_runs, model = test_runs(sampler, n_cpu, "strategy_wilcoxon", ds)
      ttime = ttime + _ttime

    if verbose: print "strategy_wilcoxon, pick degree: {}".format(d)

  if not found and d:
    ttime = ttime + sim_with_degree(sampler, n_cpu, "strategy_wilcoxon", d, n_runs)

  return ttime


def simulate(data, n_cpu, strategy, b):
  global degrees
  degrees = sorted(data[b].keys())
  sampler = partial(sampling, data, b)
  res = []
  for i in xrange(301):
    res.append(strategy(sampler, n_cpu))
  return res


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
  parser.add_option("-d", "--dir",
    action="store", dest="data_dir", default="data",
    help="output folder")
  parser.add_option("-b", "--benchmark",
    action="append", dest="benchmarks", default=[],
    help="benchmark(s) of interest")
  parser.add_option("--all",
    action="store_true", dest="all_strategies", default=False,
    help="simulate *all* modeled strategies")
  parser.add_option("-v", "--verbose",
    action="store_true", dest="verbose", default=False,
    help="verbosely print out simulation data")

  (opt, args) = parser.parse_args()

  global verbose
  verbose = opt.verbose

  db = PerfDB(opt.user, opt.db)
  db.drawing = True
  db.detail_space = True
  db.calc_stat(opt.benchmarks, True, opt.eid)
  data = db.raw_data

  merged = util.merge_succ_fail(data, 1000)

  n_cpu = 32
  _simulate = partial(simulate, merged, n_cpu)
  simulators = {}
  simulators["wilcoxon"] = partial(_simulate, strategy_wilcoxon)
  if opt.all_strategies:
    simulators["random"] = partial(_simulate, strategy_random)
    strategy_min_time = partial(strategy_time, min, "strategy_min_time")
    strategy_max_time = partial(strategy_time, max, "strategy_max_time")
    simulators["min(time)"] = partial(_simulate, strategy_min_time)
    simulators["max(time)"] = partial(_simulate, strategy_max_time)

  for b in merged:
    print "\n=== benchmark: {} ===".format(b)

    _simulators = simulators.copy()
    if opt.all_strategies:
      degrees = sorted(merged[b].keys())
      for d in degrees:
        strategy_fixed_d = partial(strategy_fixed, d)
        _simulators["fixed({})".format(d)] = partial(_simulate, strategy_fixed_d)

    for s in sorted(_simulators.keys()):
      res = _simulators[s](b)
      s_q = " | ".join(map(str, util.calc_percentile(res)))
      print "{} : {} ({}) [ {} ]".format(s, np.mean(res), np.var(res), s_q)


if __name__ == "__main__":
  sys.exit(main())

