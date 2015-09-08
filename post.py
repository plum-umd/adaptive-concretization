#!/usr/bin/env python

import glob
import os
from optparse import OptionParser
import pprint
import re
import sys

fvname_re = re.compile(r"\S+/(\S+)_vanilla_\d+\.txt")
fpname_re = re.compile(r"\S+/(\S+)_parallel_core(\d+)_(\S+)_\d+_\d+\.txt")
fsname_re = re.compile(r"\S+/(\S+)_single_(\d+)_.+\.txt")

def find_config(output, single):
  m = re.search(fsname_re if single else fpname_re, output)
  if m:
    benchmark = m.group(1)
    if single:
      degree = int(m.group(2))
      core = 1
      strategy = "FIXED"
    else:
      core = int(m.group(2))
      strategy = m.group(3)
      if "FIXED" in strategy:
        degree = int(strategy.split('_')[-1])
        strategy = "FIXED"
      else:
        degree = None

    return (benchmark, strategy, core, degree)

  else:
    m = re.search(fvname_re, output)
    if m:
      benchmark = m.group(1)
      return (benchmark, "VANILLA", 1, 0)

    else:
      return (None, None, None, None)


f_trial_re = re.compile(r"parallel trial.* \((\d+)\) failed")
s_trial_re = re.compile(r"resolved within (\d+) complete parallel trial")
ttime_re = re.compile(r"Total time = ([+|-]?(0|[1-9]\d*)(\.\d*)?([eE][+|-]?\d+)?)")
deg_re = re.compile(r"degree choice: (\d+)")
lucky_re = re.compile(r"lucky \(degree: (\d+)\)")

be_separator_re = re.compile(r"=== parallel trial.* \((\d+)\) (\S+) ===")


def analyze(output, b, s, c, d):
  run_record = {}
  run_record["benchmark"] = b
  run_record["strategy"] = s
  run_record["core"] = c

  with open(output, 'r') as f:
    degree = d
    f_trial = -1
    s_trial = -1
    ttime = -1
    f_times = []
    s_times = []
    lines = []
    for line in f:
      ## information from front-end
      m = re.search(f_trial_re, line)
      if m:
        _f_trial = int(m.group(1))
        f_trial = max(f_trial, _f_trial)

      else:
        m = re.search(s_trial_re, line)
        if m:
          s_trial = int(m.group(1))

      m = re.search(deg_re, line)
      if m:
        degree = int(m.group(1))

      else:
        m = re.search(lucky_re, line)
        if m:
          degree = int(m.group(1))

      m = re.search(ttime_re, line)
      if m:
        ttime = int(float(m.group(1)))

      ## information from back-end
      m = re.search(be_separator_re, line)
      if m:
        if m.group(2) in ["failed", "solved"]:
          record = be_analyze_lines(lines, b, s, degree)
          if record["succeed"] == "Succeed":
            s_times.append(record["ttime"])
          else: # "Failed"
            f_times.append(record["ttime"])

          lines = []

      else:
        lines.append(line)

    # for plain Sketch, the whole message is from back-end
    if s == "VANILLA":
      record = be_analyze_lines(lines, b, s, degree)
      if record["succeed"] == "Succeed":
        s_times.append(record["ttime"])
      else: # "Failed"
        f_times.append(record["ttime"])

    run_record["degree"] = degree
    trial = len(f_times) + len(s_times)
    run_record["trial"] = trial
    s_succeed = "Succeed" if any(s_times) else "Failed"
    run_record["succeed"] = s_succeed
    f_time_sum = sum(f_times)
    s_time_sum = sum(s_times)
    run_record["ttime"] = ttime
    run_record["stime"] = s_time_sum
    run_record["ftime"] = float(f_time_sum) / len(f_times) if f_times else 0
    run_record["ctime"] = f_time_sum + s_time_sum

  return run_record


exit_re = re.compile(r"Solver exit value: ([-]?\d+)")
be_tout_re = re.compile(r"timed out: (\d+)")
time_re = "([+|-]?(0|[1-9]\d*)(\.\d*)?([eE][+|-]?\d+)?)"
be_etime_re = re.compile(r"elapsed time \(s\) .* {}".format(time_re))
be_stime_re = re.compile(r"Total elapsed time \(ms\):\s*{}".format(time_re))
be_ttime_re = re.compile(r"TOTAL TIME {}".format(time_re))

propg_re = re.compile(r"f# %assign: .* propagated: (\d+)")
seed_re = re.compile(r"SOLVER RAND SEED = (\d+)")

odds_re = re.compile(r"(H__\S+) odds = 1/(\d+)")
range_re = re.compile(r"(H__\S+) .+ bnd= (\d+)")

harness_re = re.compile(r"before  EVERYTHING: (\S+)__.*")
nodes_re = re.compile(r"Final Problem size: Problem nodes = (\d+)")


def be_analyze_lines(lines, b, s, d):
  run_record = {}
  run_record["benchmark"] = b
  run_record["strategy"] = s
  run_record["degree"] = d
  run_record["dag"] = []
  run_record["hole"] = []

  exit_code = -1
  etime = 0
  ttime = 0
  timeout = None
  succeed = False
  propagation = -1
  for line in reversed(lines):
    m = re.search(exit_re, line)
    if m:
      exit_code = int(m.group(1))
      succeed |= exit_code == 0
    if "ALL CORRECT" in line:
      succeed |= True

    m = re.search(be_ttime_re, line)
    if m:
      ttime = ttime + int(float(m.group(1)))
    m = re.search(be_stime_re, line)
    if m:
      etime = float(m.group(1))
    m = re.search(be_tout_re, line)
    if m:
      timeout = int(m.group(1))
    m = re.search(propg_re, line)
    if m:
      propagation = int(m.group(1))
      break

  for line in lines:
    m = re.search(be_etime_re, line)
    if m:
      etime = int(float(m.group(1)) * 1000)
      break

  s_succeed = "Succeed" if succeed else "Failed"
  run_record["succeed"] = s_succeed
  if timeout: _time = timeout
  elif etime: _time = etime
  else: _time = ttime
  run_record["ttime"] = _time
  run_record["propagation"]= propagation

  hole = ""
  odds = -1
  hole_r = ""
  size = -1
  harness = ""
  for line in lines:

    m = re.search(odds_re, line)
    if m:
      hole = m.group(1)
      odds = int(m.group(2))

    m = re.search(range_re, line)
    if m:
      hole_r = m.group(1)
      size = max(size, int(m.group(2)))
      h_record = {"replaced": "Replaced", "name": hole, "odds": odds, "size": size}
      run_record["hole"].append(h_record)
      hole = ""
      odds = -1
      hole_r = ""
      size = -1

    m = re.search(harness_re, line)
    if m:
      harness = m.group(1)

    m = re.search(nodes_re, line)
    if m and harness:
      dag = int(m.group(1))
      run_record["dag"].append( {"harness": harness, "size": dag} )
      harness = ""

    m = re.search(seed_re, line)
    if m:
      seed = int(m.group(1))
      run_record["seed"] = seed

  return run_record


def be_analyze(output, b, s, d):
  lines = []
  with open(output, 'r') as f:
    lines = f.readlines()
  record = be_analyze_lines(lines, b, s, d)
  del record["strategy"] # table RunS doesn't have this field
  return record


def main():
  parser = OptionParser(usage="usage: %prog [options] (output_file)*")
  parser.add_option("-b", "--benchmark",
    action="append", dest="benchmarks", default=[],
    help="benchmark(s) under analysis")
  parser.add_option("-d", "--dir",
    action="store", dest="data_dir", default="data",
    help="output folder")
  parser.add_option("-s", "--single",
    action="store_true", dest="single", default=False,
    help="analyze backend behavior from single threaded executions")

  (opt, args) = parser.parse_args()

  if not args:
    outputs = glob.glob(os.path.join(opt.data_dir, "*.txt"))
    # filter out erroneous cases (due to broken pipes, etc.)
    outputs = filter(lambda f: os.path.getsize(f) > 0, outputs)
  else:
    outputs = args

  pp = pprint.PrettyPrinter(indent=2)

  for output in outputs:
    b, s, c, d = find_config(output, opt.single)
    if b:
      if opt.benchmarks and b not in opt.benchmarks: continue

      if opt.single:
        record = be_analyze(output, b, s, d)
      else:
        record = analyze(output, b, s, c, d)

      pp.pprint(record)


if __name__ == "__main__":
  sys.exit(main())

