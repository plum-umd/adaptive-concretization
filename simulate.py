#!/usr/bin/env python

from collections import Counter
from itertools import repeat, chain
from functools import partial
import random
from optparse import OptionParser
import sys

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon

from db import PerfDB
import util

# verbosity
verbose = False

# accumulated total time
g_ttime = 0
ttime_max = 7200000 # 2 hours

# p-value
g_pVal = 0.2

# default-ish degrees
degrees = [16, 32, 64, 128, 512, 1024, 2048, 4096]

# absent degrees
abs_degrees = []

model = {}
sMap = {}

def sampling(data, b, d, n=1):
  res = {}
  if d not in data[b]:
    abs_degrees.append(d)
    return res

  pop = filter(lambda i: not data[b][d]["ttime"][i][0], range(len(data[b][d]["search space"])))
  try:
    pop_len = len(data[b][d]["search space"])
    #if verbose: print "Sampling from {} trials for degree {}...".format(pop_len, d)
    samp = random.sample(pop, n)
  except ValueError: # sample larger than population
    samp = list(chain.from_iterable(repeat(range(pop_len), n / pop_len))) + random.sample(range(pop_len), n % pop_len) 
  except TypeError: # single value
    samp = repeat(pop_len, n)

  res["p"] = data[b][d]["p"]
  for k in ["ttime", "search space"]:
    res[k] = map(lambda idx: data[b][d][k][idx], samp)
  #if verbose: print "data: {}".format(res)
  return res


def sim_with_degree(sampler, n_cpu, s_name, d, n_runs = 0):
  ttime = 0
  found_d = -1
  while found_d < 0 and ttime <= ttime_max:
    runs = sampler(d, n_cpu)
    _n_runs = 0
    _ttime = 0
    for s, t in runs["ttime"]:
      _n_runs = _n_runs + 1
      _ttime = _ttime + t
      if s:
        found_d = d
        if verbose: print "{} found a solution within {} trials".format(s_name, n_runs+_n_runs)
        _ttime = t
        break
    n_runs = n_runs + _n_runs
    ttime = ttime + _ttime

  return found_d, ttime


def run_async_trials(sampler, d, n, s_name):
  _soltime = 0
  _ttime = 0
  found_d = -1
  ttime = 0
  t_runs = sampler(d, n)
  if not t_runs:
    return ttime, found_d, len(t_runs)

  if d not in model:
    model[d] = {}
    model[d]["runs"] = []
    model[d]["ttime"] = 0
    model[d]["search space"] = []

  for s, t in t_runs["ttime"]:
    model[d]["runs"].append(t)
    if found_d > 0:
      if s and t < _soltime: _soltime = t
    elif s:
      found_d = d
      _soltime = t
    _ttime = _ttime + t

  if found_d > 0:
    if verbose: print "One of {} async trials found a solution.".format(s_name, len(t_runs["ttime"]))
    ttime = _soltime
  else:
    ttime = (_ttime / len(t_runs["ttime"]))

  model[d]["p"] = t_runs["p"]
  model[d]["search space"] = model[d]["search space"] + t_runs["search space"]
  return ttime, found_d, len(t_runs)


def test_runs(sampler, n_cpu, s_name, ds):
  ttime = 0
  found_d = -1
  n_runs = 0
  for d in ds:
    if d not in model:
      model[d] = {}
      model[d]["runs"] = []
      model[d]["ttime"] = 0
      model[d]["search space"] = []
    l = len(model[d]["runs"])
    if l >= n_cpu/2: break
    #if verbose: print "{} trials exist! Run {} trials more!".format(l, n_cpu/2 - l)
    t_runs = sampler(d, n_cpu/2 - l)

    _n_runs = 0
    _ttime = 0
    if not t_runs: break
    for s, t in t_runs["ttime"]:
      model[d]["runs"].append(t)
      _n_runs = _n_runs + 1
      _ttime = _ttime + t
      if s:
        found_d = d
        print "{} found a solution while {} test runs".format(s_name, n_runs+_n_runs)
        _ttime = t
        break

    n_runs = n_runs + _n_runs
    model[d]["ttime"] = model[d]["ttime"] + _ttime
    model[d]["p"] = t_runs["p"]
    model[d]["search space"] = model[d]["search space"] + t_runs["search space"]

    ttime = ttime + _ttime
    if found_d > 0: break

  return ttime, found_d, n_runs


def strategy_fixed(d, sampler, n_cpu):
  _, ttime = sim_with_degree(sampler, n_cpu, "strategy_fixed_()".format(d), d)
  return d, ttime


def strategy_random(sampler, n_cpu):
  # pick a degree randomly
  d = random.choice(degrees)
  print "strategy_random, pick degree: {}".format(d)

  _, ttime = sim_with_degree(sampler, n_cpu, "strategy_random", d)
  return d, ttime


def strategy_time(f, msg, sampler, n_cpu):
  ttime, found_d, n_runs = test_runs(sampler, n_cpu, msg, degrees)

  # resampling with likelihood degree
  if found_d < 0:
    est = []
    ds = []
    for d in model:
      est.append(model[d]["ttime"])
      ds.append(d)
    idx = est.index(f(est))
    d = ds[idx]
    print "{}, pick degree: {}".format(msg, d)

    _, _ttime = sim_with_degree(sampler, n_cpu, msg, d, n_runs)
    ttime = ttime + _ttime

  return ttime


def strategy_wilcoxon(sampler, n_cpu, sampleBnd=0):
  global g_ttime
  if sampleBnd == 0: sampleBnd = max(8, n_cpu/2) * 3
  
  def comp_dist(d):
    res = []
    if d not in model:
      if verbose: print "degree {} does not exist in {}".format(d, model.keys())
      return res
    for i in xrange(len(model[d]["runs"])):
      t = model[d]["runs"][i]
      p = model[d]["search space"][i]
      if p == 0:
        p = 10000
      res.append(t * p)
    return res

  def sampleRequested(degree):
    if degree in sMap:
      return sMap[degree];
    else:
      sMap[degree] = 0
    return sMap[degree]

  def sample(sampler, degree, s_name):
    prev = sampleRequested(degree)
    sMap[degree] = prev + n_cpu/2
    _ttime, _found_d, _n_runs = run_async_trials(sampler, degree, n_cpu/2, s_name)
    return _ttime, _found_d, _n_runs

  def compare_async(d1, d2):
    if verbose: print "Comparing {} and {}:".format(d1, d2)
    len_a = 0
    len_b = 0
    req_a = 0
    req_b = 0
    _found_d = -1
    _n_runs = 0
    global g_ttime
    while len_a < sampleBnd or len_b < sampleBnd:
      dist_a = comp_dist(d1)
      dist_b = comp_dist(d2)
      len_a = len(dist_a)
      len_b = len(dist_b)
      if _found_d > 0:
        _pvalue = 0
      elif not (dist_a and dist_b):
        _pvalue = 0
      else:
        if len(dist_a) != len(dist_b):
          if verbose: print "length mismatch: {} vs. {}".format(len(dist_a), len(dist_b))
          shorter = min(len(dist_a), len(dist_b))
          dist_a = dist_a[:shorter]
          dist_b = dist_b[:shorter]
        _rank_sum, _pvalue = wilcoxon(dist_a, dist_b)
        if verbose: print "p-value: {}".format(_pvalue)
        if _pvalue < g_pVal: break
        elif len(dist_a) >= sampleBnd and len(dist_b) >= sampleBnd: break
      req_a = sampleRequested(d1)
      req_b = sampleRequested(d2)
      if req_a >= sampleBnd and req_b >= sampleBnd: break
      if req_a <= req_b and req_a < sampleBnd:
        _ttime, _found_d, _n_runs = sample(sampler, d1, "strategy_wilcoxon")
        g_ttime = g_ttime + _ttime
        len_a = len_a + _n_runs
        req_a = sampleBnd if _n_runs == 0 else req_a + _n_runs
      if req_b <= req_a and req_b < sampleBnd:
        _ttime, _found_d, _n_runs = sample(sampler, d2, "strategy_wilcoxon")
        g_ttime = g_ttime + _ttime
        len_b = len_b + _n_runs
        req_b = sampleBnd if _n_runs == 0 else req_b + _n_runs

    return dist_a, dist_b, _found_d, _n_runs, _pvalue

  def compare_single(d1, d2):
    if verbose: print "Comparing degrees {} and {}:".format(d1, d2)
    _ttime, _found_d, _n_runs = test_runs(sampler, n_cpu, "strategy_wilcoxon", [d1, d2])
    g_ttime = g_ttime + _ttime
    dist_d1 = comp_dist(d1)
    dist_d2 = comp_dist(d2)
    if _found_d > 0:
      _pvalue = 0
    elif not (dist_d1 and dist_d2):
      _pvalue = 0
    elif len(dist_d1) != len(dist_d2):
      if verbose: print "length mismatch: {} vs. {}".format(len(dist_d1), len(dist_d2))
      _pvalue = 0
    else: _rank_sum, _pvalue = wilcoxon(dist_d1, dist_d2)
    if verbose: print "p-value: {}".format(_pvalue)
    return dist_d1, dist_d2, _found_d, _n_runs, _pvalue

  def binary_search(degree_l, degree_h, cmpr):
    if degree_l == degree_h: return degree_l
    dist_l, dist_h, found_d, n_runs, pvalue = cmpr(degree_l, degree_h)
    if pvalue == 0:
      return [degree_l, degree_h]
    elif degree_h - degree_l <= degrees[0]:
      mean_l = np.mean(dist_l)
      mean_h = np.mean(dist_h)
      return degree_l if mean_l <= mean_h else degree_h
    else:
      degree_m = (degree_l + degree_h) / 2
      dist_dl, dist_dm, found, n_runs, pvalue = cmpr(degree_l, degree_m)
      if pvalue <= g_pVal: # the median diff. is significatly different
        if np.mean(dist_dl) < np.mean(dist_dm):
          return binary_search(degree_l, degree_m, cmpr)
        else:
          return binary_search(degree_m, degree_h, cmpr)
      else: return degree_m
  
  pivots = [0, 1]

  d = None
  found_d = -1
  fixed = False
  n_runs = 0
  cmpr = compare_async
  while found_d < 0 and not fixed and pivots[1] < len(degrees):
    fixed = True
    ds = [ degrees[pivot] for pivot in pivots ]
    d1, d2 = ds
    dist_d1, dist_d2, found_d, n_runs, pvalue = cmpr(d1, d2)

    if found_d > 0:
      print "strategy_wilcoxon, solution found at degree: {}".format(found_d)
      return found_d, g_ttime
    elif not dist_d1:
      pivots[0] = pivots[0] + 1
      pivots[1] = pivots[1] + 1
      fixed = False
    elif not dist_d2:
      pivots[1] = pivots[1] + 1
      fixed = False
    elif pvalue <= g_pVal: # the median diff. is significatly different
      if np.mean(dist_d1) < np.mean(dist_d2):
        # left one is better, climbing done
        break
      else:
        pivots[0] = pivots[0] + 1
        pivots[1] = pivots[1] + 1
        fixed = False
    else: # i.e., can't differentiate two degrees
      pivots[1] = pivots[1] + 1
      fixed = False

  if pivots[1] == len(degrees): pivots[1] = pivots[1] - 1

  # binary search now
  dl = degrees[pivots[0]]
  dh = degrees[pivots[1]]
  d = binary_search(dl, dh, cmpr)
  print "strategy_wilcoxon, pick degree: {}".format(d)
  if found_d < 0 and type(d) is int and g_ttime <= ttime_max:
    found, _ttime = sim_with_degree(sampler, n_cpu, "strategy_wilcoxon", d, n_runs)
    g_ttime = g_ttime + _ttime

  return d, g_ttime


def simulate(data, n_cpu, strategy, b):
  global degrees
  global model
  global sMap
  #degrees = sorted(data[b].keys())
  sampler = partial(sampling, data, b)
  res = []
  dgrs = {}
  ranges = []
  for i in xrange(301):
    model = {}
    sMap = {}
    _d, _ttime = strategy(sampler, n_cpu)
    res.append(_ttime)
    if type(_d) is int: # i.e., fixed single degree
      if _d in dgrs: dgrs[_d] = dgrs[_d] + 1
      else: dgrs[_d] = 1
    elif type(_d) is list: # i.e., a range of degrees
      ranges.append(_d)
      low_choice = ((_d[0]-1) / degrees[0] + 1) * degrees[0]
      high_choice = _d[1] / degrees[0] * degrees[0]
      choices = range(low_choice, high_choice+1, degrees[0])
      for i in choices:
        if i in dgrs: dgrs[i] = dgrs[i] + (1 / (len(choices) * 1.0))
        else: dgrs[i] = 1 / (len(choices) * 1.0)
  print "{} simulations ({}%) found fixed degrees!".format(301 - len(ranges), (301-len(ranges))/3.01)
  #for [low, high] in ranges:
  #  for dgr in dgrs:
  #    if low <= dgr and dgr <= high: dgrs[dgr] = dgrs[dgr] + 1
  pop = []
  for d in sorted(dgrs.keys()):
    est = "N/A"
    if d in data[b]:
      idxs = filter(lambda i: not data[b][d]["ttime"][i][0], range(len(data[b][d]["search space"])))
      est = np.mean(map(lambda i: data[b][d]["ttime"][i][1] * ((data[b][d]["search space"][i] - 1) / n_cpu + 1), idxs))
    print "degree {}: {} times (estimated time: {})".format(d, dgrs[d], est)
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
    action="append", dest="eids", type="int", default=[],
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
  parser.add_option("-p", "--p-value",
    action="store", dest="p_value", type="float", default=0.2,
    help="p-value for Wilcoxon test")
  parser.add_option("-v", "--verbose",
    action="store_true", dest="verbose", default=False,
    help="verbosely print out simulation data")

  (opt, args) = parser.parse_args()

  global verbose, g_pVal
  verbose = opt.verbose
  g_pVal = opt.p_value

  db = PerfDB(opt.user, opt.db)
  db.drawing = True
  db.detail_space = True
  if not opt.eids: opt.edis = [11]
  db.calc_stat(opt.benchmarks, True, opt.eids)
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
      print "Simulating strategy {}...".format(s)
      res = _simulators[s](b)
      print "{} simulations done.".format(len(res))
      s_q = " | ".join(map(str, util.calc_percentile(res)))
      print "{} : {} ({})\n\t[ {} ]".format(s, np.mean(res), np.var(res), s_q)

  print "absent degrees: {}".format(Counter(abs_degrees))


if __name__ == "__main__":
  sys.exit(main())


