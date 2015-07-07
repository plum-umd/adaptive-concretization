#!/usr/bin/env python

from optparse import OptionParser
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import util
from db import PerfDB

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

      # figure per benchmark and key
      png = os.path.join(out_dir, "{}_{}.png".format(b,k))
      print "drawing ", png
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

  plt.legend(loc="best")

  png = os.path.join(out_dir, "vee.png")
  print "drawing ", png
  plt.savefig(png)
  plt.close()


def fig_parallel(data, out_dir):
  # { benchmark: { strategy: { #core: { k: v ... } } } }
  pass

def main():
  parser = OptionParser(usage="usage: %prog [options]")
  parser.add_option("--user",
    action="store", dest="user", default="sketchperf",
    help="user name for database")
  parser.add_option("--db",
    action="store", dest="db", default="concretization",
    help="database name")
  parser.add_option("-e", "--eid",
    action="store", dest="eid", type="int", default=0,
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

  (opt, args) = parser.parse_args()

  db = PerfDB(opt.user, opt.db)
  db.drawing = True
  db.calc_stat(opt.benchmarks, opt.single, opt.eid)
  data = db.raw_data

  if opt.single:
    fig_single(data, opt.data_dir)
  else:
    fig_parallel(data, opt.data_dir)


if __name__ == "__main__":
  sys.exit(main())

