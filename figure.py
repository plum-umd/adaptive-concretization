#!/usr/bin/env python

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from db import PerfDB

DATA="data"

def main():
  db = PerfDB()
  db.drawing = True
  db.calc_stat()
  data = db.raw_data

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

      png = os.path.join(DATA, "{}_{}.png".format(b,k))
      print png
      plt.savefig(png)
      plt.close()


if __name__ == "__main__":
  sys.exit(main())

