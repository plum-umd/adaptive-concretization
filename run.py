#!/usr/bin/env python

import datetime
import errno
import json
import math
import multiprocessing
import os
from optparse import OptionParser
import subprocess
import sys
import shutil
import tempfile


BENCHMARK = "benchmark"
DATA = "data"


def run(b, path, main, seed):
  res = False
  try:
    tmp_dir = tempfile.mkdtemp()
    opts = []
    opts.extend(["-V", "5"])
    opts.extend(["--slv-seed", str(seed)])
    opts.extend(["--fe-output", tmp_dir])
    opts.extend(["--fe-inc", path])
    cmd = ["sketch"] + opts + [main]

    s_cmd = ' '.join(cmd)
    print "running: {}".format(s_cmd)

    output = os.path.join(DATA, "{}_vanilla_{}.txt".format(b, seed))
    with open(output, 'w') as f:
      try:
        subprocess.check_call(cmd, stdout=f)
        res = True
      except subprocess.CalledProcessError: pass

  finally:
    try:
      shutil.rmtree(tmp_dir)
    except OSError as exc:
      if exc.errno != errno.ENOENT: raise

  return res


def p_run(b, path, main, repeat):
  n_cpu = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(min(repeat, n_cpu))
  now = int(datetime.datetime.now().strftime("%H%M%S"))
  seed = now * (10 ** int(math.log(repeat, 10)))

  results = []
  try:
    for i in xrange(repeat):
      r = pool.apply_async(run, (b, path, main, abs(seed+i)), )
      results.append(r)
    pool.close()
  except KeyboardInterrupt:
    pool.close()
    pool.terminate()
  except AssertionError:
    pass
  finally:
    pool.join()

  for i in xrange(repeat):
    try:
      r = results[i].get(1)
    except IndexError: pass
    except multiprocessing.TimeoutError: pass

  return 0


def get_datetime():
  return datetime.datetime.now().strftime("%y%m%d_%H%M%S")

g_opt = None

def fe_p_run(b, path, main, strategy, core, degree=None):
  opts = []
  opts.extend(["-V", "5"])
  opts.extend(["--slv-parallel"])
  opts.extend(["--slv-randassign"])
  opts.extend(["--slv-strategy", strategy])
  opts.extend(["--slv-p-cpus", str(core)])
  opts.extend(["--slv-p-trials", str(32*32*3)])
  if degree:
    opts.extend(["--slv-randdegree", degree])
  if g_opt.timeout:
    opts.extend(["--slv-timeout", str(g_opt.timeout)])
  if g_opt.ntimes:
    opts.extend(["--slv-ntimes", str(g_opt.ntimes)])

  if "sygus" in path:
    opts.extend(["--fe-def", "BND=5"])
    opts.extend(["--bnd-unroll-amnt", '64'])

  opts.extend(["--fe-inc", path])
  cmd = ["sketch"] + opts + [main]

  s_cmd = ' '.join(cmd)
  print "running: {}".format(s_cmd)

  _strategy = strategy
  if strategy == "NOT_SET":
    _strategy = "FIXED_{}".format(degree)

  output = os.path.join(DATA, "{}_parallel_core{}_{}_{}.txt".format(b, core, _strategy, get_datetime()))
  with open(output, 'w') as f:
    try:
      subprocess.check_call(cmd, stdout=f)
    except subprocess.CalledProcessError:
      pass

  return 0


def be_p_run(b, path, main, degree):
  opts = []
  opts.extend(["-V", "5"])
  opts.extend(["--slv-randassign"])
  opts.extend(["--slv-randdegree", degree])
  opts.extend(["--slv-mem-limit", str(4*1024*1024*1024)])
  opts.extend(["--fe-inc", path])

  # custom CEGIS wrapper that runs backend in parallel
  pwd = os.path.dirname(os.path.realpath(__file__))
  cegis = os.path.join(pwd, "psketch.py")
  opts.extend(["--fe-cegis-path", cegis])

  # to store single backend run's output in a distinct file
  # trick to pass arguments to the wrapper
  opts.append("--be:conc-benchmark={}".format(b))
  opts.append("--be:conc-repeat={}".format(g_opt.repeat))
  opts.append("--be:conc-timeout={}".format(g_opt.timeout))
  opts.append("--be:conc-register={}".format(g_opt.register))

  cmd = ["sketch"] + opts + [main]

  s_cmd = ' '.join(cmd)
  print "running: {}".format(s_cmd)

  try:
    subprocess.check_call(cmd)
  except subprocess.CalledProcessError:
    pass # ignore, since backend may fail

  return 0


def main():
  parser = OptionParser(usage="usage: %prog [options]")
  parser.add_option("--config",
    action="store", dest="config", default="config.json",
    help="configuration")
  parser.add_option("-b", "--benchmark",
    action="append", dest="benchmarks", default=[],
    help="benchmark(s) under test")
  parser.add_option("--strategy",
    action="append", dest="strategies", default=[],
    help="strategies of interest")
  parser.add_option("-c", "--core",
    action="append", dest="cores", default=[],
    help="# of cores to use")
  parser.add_option("-d", "--degree",
    action="append", dest="degrees", default=[],
    help="randdegree(s) of interest")
  parser.add_option("-r", "--repeat",
    action="store", dest="repeat", type="int", default=1,
    help="# of experiments to be conducted")
  parser.add_option("--timeout",
    action="store", dest="timeout", type="int", default=30,
    help="Sketch timeout (min)")
  parser.add_option("--ntimes",
    action="store", dest="ntimes", type="int", default=None,
    help="number of rounds on a single back-end invocation")
  parser.add_option("-s", "--single",
    action="store_true", dest="single", default=False,
    help="single threaded execution with various random degree and verbose outputs")
  parser.add_option("--register",
    action="store_true", dest="register", default=False,
    help="run & register at the same time (to save HDD space)")
  parser.add_option("--vanilla",
    action="store_true", dest="vanilla", default=False,
    help="run vanilla Sketch (in parallel, with different seeds)")

  (opt, args) = parser.parse_args()
  global g_opt
  g_opt = opt

  if opt.single:
    opt.strategies.append("SINGLE")

  # if not specified, run all benchmarks under benchmark/ folder
  if not opt.benchmarks:
    opt.benchmarks = os.listdir(BENCHMARK)

  config = json.load(open(opt.config))

  for b in opt.benchmarks:
    if b not in config: continue
    path = os.path.join(BENCHMARK, b)
    main = os.path.join(path, config[b]["main"])

    strategies = opt.strategies if opt.strategies else config[b]["strategies"]
    cores = opt.cores if opt.cores else config[b]["cores"]
    degrees = opt.degrees if opt.degrees else config[b]["degrees"]

    if opt.vanilla: # run vanilla Sketch in parallel
      p_run(b, path, main, opt.repeat)

    else: # use sketch-frontend's parallel running
      for i in xrange(opt.repeat):
        for strategy in strategies:

          if strategy == "SINGLE": # parallel running via wrapper
            for degree in degrees:
              r_bak = opt.repeat
              if "single_repeat" in config[b]:
                opt.repeat = config[b]["single_repeat"]

              be_p_run(b, path, main, str(degree))

              opt.repeat = r_bak

          elif strategy == "NOT_SET": # fixed degree
            for core in cores:
              for degree in degrees:
                fe_p_run(b, path, main, strategy, core, str(degree))

          elif strategy == "WILCOXON": # adaptive concretization
            for core in cores:
              fe_p_run(b, path, main, strategy, core)

  return 0


if __name__ == "__main__":
  sys.exit(main())

