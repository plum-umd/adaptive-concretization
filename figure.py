#!/usr/bin/env python

from optparse import OptionParser
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import util
from db import PerfDB


verbose = False

def fig_single(data, out_dir):
  # { benchmark: [(m, siqr)_d1, (m, siqr)_d2, ...], ... }
  e_ts = {}

  # { benchmark: { degree: { k: v ... } } }
  for b in data:
    _e_ts = {}

    _keys = set([])
    for d in data[b]:
      for k in data[b][d]:
        if type(data[b][d][k]) in [list, dict]: _keys.add(k)

      if "E(t)" in data[b][d]:
        _e_ts[d] = data[b][d]["E(t)"]

    # normalizing
    min_e_ts = min([m for (m, _) in _e_ts.values()])
    e_ts[b] = {}
    for (d, (m, siqr)) in _e_ts.iteritems():
      if m == float("inf"):
        e_ts[b][d] = (1000, 0)
      else:
        e_ts[b][d] = (m/min_e_ts, siqr/m)

    for k in _keys:
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.set_xscale("log", nonposx="clip")
      ax.set_xlabel("degree")
      if k in ["Succeed", "Failed"]:
        ax.set_ylabel("running time (ms)")
      else:
        ax.set_ylabel(k)
      plt.title(b)

      for d in data[b]:
        if k in data[b][d]:
          x = [int(d)] * len(data[b][d][k])
          ax.scatter(x, data[b][d][k])
          if verbose: print d, data[b][d][k]

      # figure per benchmark and key
      png = os.path.join(out_dir, "{}_{}.png".format(b,k))
      print "drawing", png
      plt.savefig(png)
      plt.close()

  # vee chart
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_xscale("log", nonposx="clip")
  ax.set_xlabel("degree")
  ax.set_yscale("log", nonposx="clip")
  ax.set_ylabel("expected running time (normalized)")

  colors = "bgrcmykw"
  color_index = 0

  for b in e_ts:
    # not sorted...
    #xs = e_ts[b].keys()
    #ys, es = util.split(e_ts[b].values())

    xs = []
    ys = []
    es = []
    for d in sorted(e_ts[b].keys()):
      xs.append(d)
      m, siqr = e_ts[b][d]
      ys.append(m)
      es.append(siqr)

    #ax.errorbar(xs, ys, yerr=es, label=b, fmt="-o", color=colors[color_index])
    ax.plot(xs, ys, colors[color_index]+"o-", label=b)
    color_index += 1
    if verbose:
      print b
      print xs
      print ys
      print es

  plt.legend(loc="best")

  png = os.path.join(out_dir, "vee.png")
  print "drawing", png
  plt.savefig(png)
  plt.close()


def fig_parallel(data, out_dir, invert):
  # { benchmark: { strategy: (m, siqr) } }
  bsv = {}

  # { benchmark: { strategy: { #core: { k: v ... } } } }
  for b in data:
    bsv[b] = {}
    for s in data[b]:
      for c in data[b][s]:
        ttimes = data[b][s][c]["TTIME"]
        m, siqr = util.calc_siqr(ttimes)
        if len(ttimes) <  int(13 * 0.50): m = m * 1000
        if len(ttimes) <= int(13 * 0.75): siqr = m
        bsv[b][s+str(c)] = (m, siqr)


  ## plain Sketch vs. AC
  fig = plt.figure()
  ax = fig.add_subplot(111)
  if invert:
    ax.set_ylabel("speedup")
  else:
    ax.set_ylabel("running time (normalized)")

  xs = []
  ys = []
  es = []
  for b in bsv:
    try:
      base, _ = bsv[b]["VANILLA1"]
    except KeyError: # e.g., bsv[menu_demo][VANILLA1] = OOM
      base, _ = bsv[b]["WILCOXON1"]
      base = base * 32

    xs.append(b)
    m, siqr = bsv[b]["WILCOXON32"]
    if invert:
      speedup = min(15, base / m)
      ys.append(speedup)
      es.append(siqr / m)
    else:
      slowdown = max(1/15, m / base)
      ys.append(slowdown)
      es.append(siqr / base)

  ys_sorted, xs_sorted = util.sort_both(ys, xs)
  ys_sorted, es_sorted = util.sort_both(ys, es)
  if verbose:
    print xs_sorted
    print ys_sorted
    print es_sorted

  xr = np.arange(len(xs))
  bars = plt.bar(xr, ys_sorted, yerr=es_sorted, ecolor='k', align="center")
  for i, ys in enumerate(ys_sorted):
    if invert:
      if ys < 0.9: bars[i].set_color('r')
      elif ys >= 2.0: bars[i].set_color('g')
      elif ys >= 1.1: bars[i].set_color("yellowgreen")
      else: bars[i].set_color('y')
    else:
      if ys > (1/0.9): bars[i].set_color('r')
      elif ys <= (1/2.0): bars[i].set_color('g')
      elif ys <= (1/1.1): bars[i].set_color("yellowgreen")
      else: bars[i].set_color('y')

  plt.xticks(xr, xs_sorted, rotation="vertical")

  plt.axhline(y=1.0, color='m')
  plt.axis("tight")
  plt.tight_layout()

  if invert:
    png_name = "sketch-ac-inv.png"
  else:
    png_name = "sketch-ac.png"
  png = os.path.join(out_dir, png_name)
  print "drawing", png
  plt.savefig(png)
  plt.close()


  ## scalability: core 1 vs. 4 vs. 32
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_ylabel("running time (normalized)")

  xs = []
  y32s = []
  e32s = []
  y4s = []
  e4s = []
  y1s = []
  e1s = []
  for b in bsv:
    try:
      base, _ = bsv[b]["VANILLA1"]
    except KeyError: # e.g., bsv[menu_demo][VANILLA1] = OOM
      base, _ = bsv[b]["WILCOXON1"]
      base = base * 32

    y32, e32 = bsv[b]["WILCOXON32"]
    y4, e4 = bsv[b]["WILCOXON4"]
    y1, e1 = bsv[b]["WILCOXON1"]

    base = max(base, y32, y4, y1)

    xs.append(b)

    y32s.append(y32 / base)
    e32s.append(e32 / base)

    if y32 <= y4:
      y4s.append((y4 / base) - (y32 / base))
    else:
      y4s.append(0)
    e4s.append(e4 / base)

    if y4 <= y1:
      y1s.append((y1 / base) - (y4 / base))
    else:
      y1s.append(0)
    e1s.append(e1 / base)

  y1bs = [ y32 + y4 for (y32, y4) in zip(y32s, y4s) ]
  if verbose:
    print xs
    print y32s, e32s
    print y4s, e4s
    print y1s, e1s

  xr = np.arange(len(xs))
  p1 = plt.bar(xr, y32s, color='g', yerr=e32s, ecolor='r', align="center")
  p2 = plt.bar(xr, y4s, color='y', bottom=y32s, yerr=e4s, ecolor='r', align="center")
  p3 = plt.bar(xr, y1s, color='m', bottom=y1bs, yerr=e1s, ecolor='r', align="center")

  plt.xticks(xr, xs, rotation="vertical")

  plt.legend( (p1[0], p2[0], p3[0]), ("32", "4", "1"), loc="best")

  plt.axis("tight")
  plt.tight_layout()

  png = os.path.join(out_dir, "scalability.png")
  print "drawing", png
  plt.savefig(png)
  plt.close()


def fig_pfunc(degree, out_dir):
  font = {"family": "serif", "size": 18}
  plt.rc("font", **font)
  label_font = {"size": 22}

  ## discontinuous p function
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_xlabel("Influence", **label_font)
  ax.set_ylabel("Concretization Probability", **label_font)
  plt.title("Degree {}".format(degree), **label_font)
  plt.ylim(ymin=0, ymax=1)

  def discontinuous(d, n):
    if n <= 0: return 0.0
    elif n > 1500: return 1.0
    else: return 1.0/max(2.0, float(d)/n)

  v_discontinuous = np.vectorize(discontinuous)

  x = np.linspace(0, 2000)
  y = v_discontinuous(degree, x)
  plt.plot(x, y, 'g', linewidth=4)

  plt.axis("tight")
  plt.tight_layout()

  png = os.path.join(out_dir, "p-discontinuous.png")
  print "drawing", png
  plt.savefig(png)
  plt.close()

  ## smooth p function
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_xlabel("Influence", **label_font)
  ax.set_ylabel("Concretization Probability", **label_font)
  plt.title("Degree {}".format(degree), **label_font)
  plt.ylim(ymin=0, ymax=1)

  def smooth(d, n):
    return (1 / (1 + np.exp(-n/d)) - 0.5) * 2

  x = np.linspace(0, 2000)
  y = smooth(degree, x)
  plt.plot(x, y, 'g', linewidth=4)

  plt.axis("tight")
  plt.tight_layout()

  png = os.path.join(out_dir, "p-smooth.png")
  print "drawing", png
  plt.savefig(png)
  plt.close()


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
  parser.add_option("-s", "--single",
    action="store_true", dest="single", default=False,
    help="refer to backend behavior from single threaded executions")
  parser.add_option("-i", "--invert",
    action="store_true", dest="invert", default=False,
    help="invert the graph, e.g., speedup instead of slowdown")
  parser.add_option("--p-func",
    action="store_true", dest="pfunc", default=False,
    help="draw probability functions")
  parser.add_option("-v", "--verbose",
    action="store_true", dest="verbose", default=False,
    help="verbosely print out data to be drawn")

  (opt, args) = parser.parse_args()

  global verbose
  verbose = opt.verbose

  if opt.pfunc:
    fig_pfunc(512, opt.data_dir)
    return 0

  db = PerfDB(opt.user, opt.db)
  db.drawing = True
  db.calc_stat(opt.benchmarks, opt.single, opt.eids)
  data = db.raw_data

  if opt.single:
    fig_single(data, opt.data_dir)
  else:
    fig_parallel(data, opt.data_dir, opt.invert)

  return 0


if __name__ == "__main__":
  sys.exit(main())

