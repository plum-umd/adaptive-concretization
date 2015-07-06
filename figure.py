#!/usr/bin/env python

from optparse import OptionParser
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from db import PerfDB


def fig_single(data, out_dir):
  # { benchmark: { degree: { k: v ... } } }
  for b in data:
    _keys = set([])
    for d in data[b]:
      for k in data[b][d]:
        if type(data[b][d][k]) in [list, dict]: _keys.add(k)

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

      png = os.path.join(out_dir, "{}_{}.png".format(b,k))
      print png
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
    pass


if __name__ == "__main__":
  sys.exit(main())

