#!/usr/bin/env python

from functools import partial
import random
from optparse import OptionParser
import sys

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon

from db import PerfDB, calc_percentile

degrees = [16, 64, 128, 512, 1024, 4096]


def stat_oracle(data, b, d, n=1):
  res = {}
  for k in data[b][d]:
    try:
      res[k] = random.sample(data[b][d][k], n)
    except TypeError: # single value
      res[k] = data[b][d][k]
  #print "oracle: {}".format(res)
  return res


def sim_with_degree(oracle, n_cpu, s_name, d, n_runs = 0):
  ttime = 0
  found = False
  while not found:
    runs = oracle(d, n_cpu)
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


def test_runs(oracle, n_cpu, s_name, ds):
  ttime = 0
  model = {}
  found = False
  n_runs = 0
  for d in ds:
    model[d] = {}
    model[d]["runs"] = []
    t_runs = oracle(d, n_cpu/2)

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
    model[d]["propagation"] = np.mean(t_runs["propagation"])
    model[d]["p"] = t_runs["p"]

    ttime = ttime + _ttime
    if found: break

  return ttime, found, n_runs, model


def strategy_fixed(d, oracle, n_cpu):
  return sim_with_degree(oracle, n_cpu, "strategy_fixed_()".format(d), d)


def strategy_random(oracle, n_cpu):
  # pick a degree randomly
  d = random.choice(degrees)
  #print "strategy_random, pick degree: {}".format(d)

  return sim_with_degree(oracle, n_cpu, "strategy_random", d)


def strategy_time(f, msg, oracle, n_cpu):
  ds = degrees[2:]
  ttime, found, n_runs, model = test_runs(oracle, n_cpu, msg, ds)

  # resampling with likelihood degree
  if not found:
    est = []
    ds = []
    for d in model:
      est.append(model[d]["ttime"])
      ds.append(d)
    idx = est.index(f(est))
    d = ds[idx]
    #print "{}, pick degree: {}".format(msg, d)

    ttime = ttime + sim_with_degree(oracle, n_cpu, msg, d, n_runs)

  return ttime


def strategy_wilcoxon(oracle, n_cpu):
  pivots = [2, len(degrees)-1]
  ds = [ degrees[pivot] for pivot in pivots ]
  ttime, found, n_runs, model = test_runs(oracle, n_cpu, "strategy_wilcoxon", ds)

  d = None
  fixed = False
  while not found and not fixed and pivots[0] < pivots[1]:
    fixed = True
    d1, d2 = ds
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
      _ttime, found, n_runs, model = test_runs(oracle, n_cpu, "strategy_wilcoxon", ds)
      ttime = ttime + _ttime

    #print "strategy_wilcoxon, pick degree: {}".format(d)

  if not found and d:
    ttime = ttime + sim_with_degree(oracle, n_cpu, "strategy_wilcoxon", d, n_runs)

  return ttime


def simulate(data, n_cpu, strategy, b):
  oracle = partial(stat_oracle, data, b)
  res = []
  for i in xrange(100):
    res.append(strategy(oracle, n_cpu))
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

  (opt, args) = parser.parse_args()

  db = PerfDB(opt.user, opt.db)
  db.drawing = True
  db.calc_stat(opt.benchmarks, True, opt.eid)
  data = db.raw_data

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

      if "Succeed" not in data[b][d]:
        _merged[b][d]["ttime"].append( (True, _max * 1000) )

      for k in data[b][d]:
        if k in ["Succeed", "Failed"]: continue
        _merged[b][d][k] = data[b][d][k]

  n_cpu = 30
  _simulate = partial(simulate, _merged, n_cpu)
  simulators = {}
  simulators["random"] = partial(_simulate, strategy_random)
  strategy_min_time = partial(strategy_time, min, "strategy_min_time")
  strategy_max_time = partial(strategy_time, max, "strategy_max_time")
  simulators["min(time)"] = partial(_simulate, strategy_min_time)
  simulators["max(time)"] = partial(_simulate, strategy_max_time)
  simulators["wilcoxon"] = partial(_simulate, strategy_wilcoxon)

  strategy_fixed_128  = partial(strategy_fixed, 128)
  strategy_fixed_512  = partial(strategy_fixed, 512)
  strategy_fixed_1024 = partial(strategy_fixed, 1024)
  strategy_fixed_4096 = partial(strategy_fixed, 4096)
  simulators["fixed(0128)"] = partial(_simulate, strategy_fixed_128)
  simulators["fixed(0512)"] = partial(_simulate, strategy_fixed_512)
  simulators["fixed(1024)"] = partial(_simulate, strategy_fixed_1024)
  simulators["fixed(4096)"] = partial(_simulate, strategy_fixed_4096)

  for b in _merged:
    print "\n=== benchmark: {} ===".format(b)
    for s in sorted(simulators.keys()):
      res = simulators[s](b)
      s_q = " | ".join(map(str, calc_percentile(res)))
      print "{} : {} ({}) [ {} ]".format(s, np.mean(res), np.var(res), s_q)


if __name__ == "__main__":
  sys.exit(main())

