#!/usr/bin/env python

from itertools import combinations
from functools import partial
import glob
from optparse import OptionParser
import operator as op
import os
import random
import sys

import mysql.connector
from mysql.connector import errorcode

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon

import post

"""
table Experiment
+-----------+
| EID | RID |
+___________+

table RunS (key: auto_inc RID)
+----------------------------------------------------------------+
| RID | Benchmark | Degree | Succeed | Seed | Propagation | Time |
+----------------------------------------------------------------+

table Dag (key: RID + Harness)
+----------------------+
| RID | Harness | Size |
+----------------------+

table Hole (key: auto_inc HID)
+-------------------------------------------+
| RID | HID | Replaced | Name | Odds | Size |
+-------------------------------------------+

table RunP (key: auto_inc RID)
+---------------------------------------------------------------------------+
| RID | Benchmark | Core | Degree | Succeed | Trial | TTime | STime | FTime |
+---------------------------------------------------------------------------+

"""

# constants
t_b = "BIT"
t_i = "INT"
t_f = "FLOAT"
t_str = "VARCHAR(255)"
t_sstr = "VARCHAR(25)"
nnull = "NOT NULL"
auto_inc = "AUTO_INCREMENT"
pkey = "PRIMARY KEY"

e_name = "Experiment"
e_schema = [
  ["EID", t_i, nnull],
  ["RID", t_i, nnull]
]

sr_name = "RunS"
sr_schema = [
  ["RID", t_i, nnull, auto_inc, pkey],
  ["BENCHMARK", t_str, nnull],
  ["DEGREE", t_i, nnull],
  ["SUCCEED", t_sstr, nnull],
  ["SEED", t_i, nnull],
  ["PROPAGATION", t_i, nnull],
  ["TTIME", t_i, nnull]
]

d_name = "Dag"
d_schema = [
  ["RID", t_i, nnull],
  ["HARNESS", t_str, nnull],
  ["SIZE", t_i, nnull],
  [pkey, "(RID, HARNESS)"]
]

h_name = "Hole"
h_schema = [
  ["RID", t_i, nnull],
  ["HID", t_i, nnull, auto_inc, pkey],
  ["REPLACED", t_sstr, nnull],
  ["NAME", t_str, nnull],
  ["ODDS", t_i, nnull],
  ["SIZE", t_i, nnull]
]

pr_name = "RunP"
pr_schema = [
  ["RID", t_i, nnull, auto_inc, pkey],
  ["BENCHMARK", t_str, nnull],
  ["STRATEGY", t_str, nnull],
  ["CORE", t_i, nnull],
  ["DEGREE", t_i, nnull],
  ["SUCCEED", t_sstr, nnull],
  ["TRIAL", t_i, nnull],
  ["TTIME", t_i, nnull],
  ["STIME", t_i, nnull],
  ["FTIME", t_f, nnull]
]

schemas = {
  e_name: e_schema,
  sr_name: sr_schema,
  d_name: d_schema,
  h_name: h_schema,
  pr_name: pr_schema
}

verbose = False

def sanitize_table_name(table_name):
  return table_name.replace('\'', '\'\'')


# benchmark => "'benchmark'"
# 100 => "100"
def quote(x):
  if type(x) in [str, unicode]: return "\'{}\'".format(x)
  else: return str(x)


# "\infty" => "\infty"
# 3.141592654 => "3.14"
def formatter(x, d=0):
  if type(x) is str: return x
  else: return "{0:,.{1}f}".format(x, d)


# ~ List.split in OCaml
# transform a list of pairs into a pair of lists
# e.g., [ (1, 'a'), (2, 'b'), (3, 'c') ] -> ([1, 2, 3], ['a', 'b', 'c'])
def split(lst):
  if not lst: return ([], [])
  else:
    try:
      a, b = zip(*lst) # zip doesn't handle an empty list
      return (list(a), list(b))
    except ValueError: # [ (1, ), (2, ) ] -> [1, 2]
      return list(zip(*lst)[0])
    except TypeError: # already 1d list
      return lst


# transform a list of pairs into a dict
def to_dict(lst):
  if not lst: return {}
  else: return { k: v for k, v in lst }


def calc_percentile(lst, ps=[0, 25, 50, 75, 100]):
  _lst = split(lst)
  a = np.array(_lst)
  f = partial(np.percentile, a)
  return map(f, ps)


def calc_siqr(lst):
  _, q1, q2, q3, _ = calc_percentile(lst)
  siqr = (q3 - q1) / 2
  return q2, siqr


class PerfDB(object):

  def __init__(self, user="sketchperf", db="concretization"):
    self.cnx = mysql.connector.connect(host="127.0.0.1", user=user)
    self.cnx.database = db # assume this db is already set up
    self.cnx.get_warnings = True
    self._drawing = False
    self._raw_data = {}

  @property
  def drawing(self):
    return self._drawing

  @drawing.setter
  def drawing(self, v):
    self._drawing = v

  def log(self, msg):
    if not self._drawing: print msg

  @property
  def raw_data(self):
    return self._raw_data

  @staticmethod
  def __execute(cur, query):
    if verbose: print "[query] {}".format(query)
    try:
      return cur.execute(query)
    except mysql.connector.Error as err:
      print "Failed to execute the previous query: {}".format(err)
      exit(1)

  # True if the table of interest indeed exists
  def table_exists(self, table_name):
    cur = self.cnx.cursor()
    query = """
      SELECT COUNT(*)
      FROM information_schema.tables
      WHERE TABLE_SCHEMA = '{0}' and TABLE_NAME = '{1}'
    """.format(self.cnx.database, sanitize_table_name(table_name))
    PerfDB.__execute(cur, query)
    ret = False
    if cur.fetchone()[0] == 1: ret = True
    cur.close()
    return ret

  # initialize the table
  def init_table(self, table_name, schema):
    cur = self.cnx.cursor()
    s_schema = ",\n\t".join(map(lambda col_spec: ' '.join(col_spec), schema))
    query = """
      CREATE TABLE IF NOT EXISTS {0} (
        {1}
      )
    """.format(sanitize_table_name(table_name), s_schema)
    PerfDB.__execute(cur, query)
    cur.close()

  # delete the table
  def drop_table(self, table_name):
    cur = self.cnx.cursor()
    query = """
      DROP TABLE IF EXISTS {0}
    """.format(sanitize_table_name(table_name))
    PerfDB.__execute(cur, query)
    cur.close()

  insert_query = """
    INSERT INTO {0} ({1}) VALUES
      {2}
  """

  # register one single-threaded run
  def __reg_run_single(self, output):
    benchmark, strategy, core, degree = post.find_config(output, True)
    if not benchmark: return
    self.log("register: {}".format(output))
    record = post.be_analyze(output, benchmark, strategy, degree)

    cur = self.cnx.cursor()
    up_call = op.methodcaller("upper")

    # register "RunS"
    _record = {}
    for k in record:
      if type(record[k]) in [list, dict]: continue
      _record[k] = record[k]

    up_keys = ','.join(map(up_call, _record.keys()))
    _values = '(' + ", ".join(map(quote, _record.values())) + ')'
    PerfDB.__execute(cur, PerfDB.insert_query.format(sr_name, up_keys, _values))
    rid = cur.lastrowid
    if not rid:
      self.log("failed to register: {}".format(output))
      cur.close()
      return None

    # register list of dictionary
    def reg_list_of_dict(table_name, lst):
      up_keys = ','.join(["RID"] + map(up_call, lst[0].keys()))
      def to_val(dic):
        return '(' + ','.join(map(quote, [rid] + dic.values())) + ')'
      _values = map(to_val, lst)
      valuess = ",\n\t".join(_values)
      PerfDB.__execute(cur, PerfDB.insert_query.format(table_name, up_keys, valuess))

    # register "Dag"
    if record["dag"]: reg_list_of_dict(d_name, record["dag"])

    # register "Hole"
    if record["hole"]: reg_list_of_dict(h_name, record["hole"])

    cur.close()
    return rid


  # register one parallel run
  def __reg_run_parallel(self, output):
    benchmark, strategy, core, degree = post.find_config(output, False)
    if not benchmark: return
    self.log("register: {}".format(output))
    record = post.analyze(output, benchmark, strategy, core, degree)

    cur = self.cnx.cursor()
    up_call = op.methodcaller("upper")

    # register "RunP"
    _record = {}
    for k in record:
      if type(record[k]) in [list, dict]: continue
      _record[k] = record[k]

    up_keys = ','.join(map(up_call, _record.keys()))
    _values = '(' + ", ".join(map(quote, _record.values())) + ')'
    PerfDB.__execute(cur, PerfDB.insert_query.format(pr_name, up_keys, _values))
    rid = cur.lastrowid
    if not rid:
      self.log("failed to register: {}".format(output))
      cur.close()
      return None

    cur.close()
    return rid


  # register the experiment
  def reg_exp(self, outputs, single, eid=0, dry=False):
    if eid <= 0:
      cur = self.cnx.cursor()
      query = """
        SELECT COALESCE(MAX(EID), 0) AS MaxEID FROM {}
      """.format(e_name)
      PerfDB.__execute(cur, query)
      eid = cur.fetchone()[0] + 1
      cur.close()

    rids = []
    for output in outputs:
      if single:
        rid = self.__reg_run_single(output)
      else:
        rid = self.__reg_run_parallel(output)
      if rid: rids.append(rid)

    if rids:
      cur = self.cnx.cursor()
      paired_rids = map(lambda rid: "({},{})".format(eid, rid), rids)
      joined_rids = ", ".join(paired_rids)
      PerfDB.__execute(cur, PerfDB.insert_query.format(e_name, "EID, RID", joined_rids))
      cur.close()

    if not dry: self.cnx.commit()


  # integrity
  def chk_integrity(self):
    # replace TTIME -1, which means timeout
    upd_query = """
      UPDATE {0}
      SET {1} = REPLACE({1}, {2}, {3})
    """.format(sr_name, "TTIME", -1, 30*60*1000) # 30 mins timeout
    cur = self.cnx.cursor()
    PerfDB.__execute(cur, upd_query)
    cur.close()


  def __select(self, funcs, cols, tables, conds, group, fetch):
    cur = self.cnx.cursor()
    _cols = []
    for col in cols:
      if funcs:
        for func in funcs:
          _cols.append("{}({})".format(func, col))
      else: _cols.append(col)
    if group:
      if group not in cols: _cols.insert(0, group)
      _group = "GROUP BY " + group
    else: _group = ""
    count_query = """
      SELECT {0}
      FROM {1}
      WHERE {2}
      {3}
    """.format(", ".join(_cols), ", ".join(tables), " AND ".join(conds), _group)
    PerfDB.__execute(cur, count_query)
    res = fetch(cur)
    cur.close()
    return res

  def __select_one(self, funcs, cols, tables, conds, group):
    _one = op.methodcaller("fetchone")
    return self.__select(funcs, cols, tables, conds, group, _one)

  def __select_all(self, funcs, cols, tables, conds, group):
    _all = op.methodcaller("fetchall")
    return self.__select(funcs, cols, tables, conds, group, _all)

  @staticmethod
  def match_EID(eid):
    return "{}.EID = {}".format(e_name, eid)

  @staticmethod
  def match_RID(table1, table2):
    return "{}.RID = {}.RID".format(table1, table2)

  @staticmethod
  def match(table_name, what, v):
    return "{}.{} = {}".format(table_name, what, quote(v))

  # statistics per benchmark and degree
  def _stat_benchmark_degree_single(self, eid, b, d, detailed=False):
    self.log("\nbenchmark: {}, degree: {}".format(b, d))

    conds = []
    conds.append(PerfDB.match_EID(eid))
    conds.append(PerfDB.match_RID(e_name, sr_name))
    conds.append(PerfDB.match(sr_name, "BENCHMARK", b))
    conds.append(PerfDB.match(sr_name, "DEGREE", d))
    n_total = self.__select_one(["COUNT"], ["*"], [e_name, sr_name], conds, None)[0]

    _conds = conds[:]
    _conds.append(PerfDB.match(sr_name, "SUCCEED", "Succeed"))
    n_succeed = self.__select_one(["COUNT"], ["*"], [e_name, sr_name], _conds, None)[0]

    self.log("success rate: {} / {}".format(n_succeed, n_total))
    self._raw_data[b][d]["p"] = float(n_succeed) / n_total

    def __stat_ttime(succeed):
      _conds = conds[:]
      _conds.append(PerfDB.match(sr_name, "SUCCEED", succeed))

      dist = self.__select_one(["MIN", "MAX", "AVG", "STDDEV"], ["TTIME"], [e_name, sr_name], _conds, None)
      _min, _max, _avg, _dev = dist
      _ttimes = self.__select_all([], ["TTIME"], [e_name, sr_name], _conds, None)
      _percentile = ""
      if _ttimes:
        self._raw_data[b][d][succeed] = split(_ttimes)
        _vals = calc_percentile(_ttimes)
        _percentile = "[" + " | ".join(map(str, _vals)) + "]"

      lucky = "lucky" if succeed == "Succeed" else "unlucky"
      self.log("{} runs' solution time: {} ({})".format(lucky, _avg, _dev))
      if _percentile: self.log("\tquantiles: {}".format(_percentile))

    __stat_ttime("Succeed")
    __stat_ttime("Failed")

    props = self.__select_all([], ["PROPAGATION"], [e_name, sr_name], conds, None)
    self._raw_data[b][d]["propagation"] = split(props)
    _percentile = calc_percentile(props)
    self.log("propagation: [{}]".format(" | ".join(map(str, _percentile))))

    _conds = conds[:]
    _conds.append(PerfDB.match_RID(sr_name, d_name))
    dag = self.__select_all(["AVG"], ["SIZE"], [e_name, sr_name, d_name], _conds, "HARNESS")
    
    self.log("average dag size:")
    for seq in dag:
      hrns, avg_sz = seq
      self.log("  {}: {}".format(hrns, avg_sz))

    if not detailed: return

    _conds = conds[:]
    _conds.append(PerfDB.match_RID(sr_name, h_name))
    hole_szs = self.__select_all([], ["SIZE"], [e_name, sr_name, h_name], _conds, "NAME")
    _, _szs = split(hole_szs)
    s_space = float(reduce(op.mul, _szs))
    self.log("search space: {}".format(s_space))

    _conds.append(PerfDB.match(h_name, "REPLACED", "Replaced"))
    rpl = self.__select_one(["COUNT"], ["*"], [e_name, sr_name, h_name], _conds, None)[0]
  
    self.log("concretization rate: {}".format(float(rpl) / n_total))

    rpl_holes = self.__select_all(["COUNT"], ["*"], [e_name, sr_name, h_name], _conds, "NAME")
    hole_dict = to_dict(rpl_holes)
    s_hole_dict = {}
    if n_succeed:
      _conds.append(PerfDB.match(sr_name, "SUCCEED", "Succeed"))
      s_holes = self.__select_all(["COUNT"], ["*"], [e_name, sr_name, h_name], _conds, "NAME")
      s_hole_dict = to_dict(s_holes)

    self.log("hole concretization histogram")
    for hole in hole_dict:
      cnt = hole_dict[hole]
      s_cnt = " ({})".format(s_hole_dict[hole]) if hole in s_hole_dict else ""
      self.log("  {}: {}{}".format(hole, cnt, s_cnt))


  def _stat_benchmark_parallel(self, eid, b, s, c):
    self.log("\nbennchmark: {}, core: {}, strategy: {}".format(b, c, s))

    conds = []
    conds.append(PerfDB.match_EID(eid))
    conds.append(PerfDB.match_RID(e_name, pr_name))
    conds.append(PerfDB.match(pr_name, "BENCHMARK", b))
    conds.append(PerfDB.match(pr_name, "CORE", c))
    conds.append(PerfDB.match(pr_name, "STRATEGY", s))
    n_total = self.__select_one(["COUNT"], ["*"], [e_name, pr_name], conds, None)[0]

    _conds = conds[:]
    _conds.append(PerfDB.match(pr_name, "SUCCEED", "Succeed"))
    n_succeed = self.__select_one(["COUNT"], ["*"], [e_name, pr_name], _conds, None)[0]

    self.log("success rate: {} / {}".format(n_succeed, n_total))

    def __stat_col(cname):
      _conds = conds[:]
      dist = self.__select_one(["MIN", "MAX", "AVG", "STDDEV"], [cname], [e_name, pr_name], _conds, None)
      _min, _max, _avg, _dev = dist
      _cols = self.__select_all([], [cname], [e_name, pr_name], _conds, None)

      _percentile = ""
      if _cols:
        self._raw_data[b][s][c][cname] = split(_cols)
        _vals = calc_percentile(_cols)
        _percentile = "[" + " | ".join(map(str, _vals)) + "]"

      self.log("{}: {} ({})".format(cname, _avg, _dev))
      if _percentile: self.log("\tquantiles: {}".format(_percentile))

    __stat_col("TRIAL")

    __stat_col("FTIME")
    __stat_col("STIME")
    __stat_col("TTIME")

    __stat_col("DEGREE")
    _conds = conds[:]
    degrees = self.__select_all(["COUNT"], ["*"], [e_name, pr_name], _conds, "DEGREE")
    d_dict = to_dict(degrees)
    s_d_dict = {}
    if n_succeed:
      _conds.append(PerfDB.match(pr_name, "SUCCEED", "Succeed"))
      s_degrees = self.__select_all(["COUNT"], ["*"], [e_name, pr_name], _conds, "DEGREE")
      s_d_dict = to_dict(s_degrees)

    self.log("degree histogram")
    for degree in sorted(d_dict.keys()):
      cnt = d_dict[degree]
      s_cnt = " ({})".format(s_d_dict[degree]) if degree in s_d_dict else ""
      self.log("  {}: {}{}".format(degree, cnt, s_cnt))


  def __get_distinct(self, col, tables, conds):
    return self.__select_all(["DISTINCT"], [col], tables, conds, None)


  # statistics per benchmark
  def _stat_benchmark(self, single, eid, b, detailed=False):
    if single:
      conds = []
      conds.append(PerfDB.match_EID(eid))
      conds.append(PerfDB.match_RID(e_name, sr_name))
      conds.append(PerfDB.match(sr_name, "BENCHMARK", b))
      degrees = self.__get_distinct("DEGREE", [e_name, sr_name], conds)
      if not degrees: return
      sorted_degrees = sorted(split(degrees))
      for d in sorted_degrees:
        self._raw_data[b][d] = {}
        self._stat_benchmark_degree_single(eid, b, d, detailed)

      ## tex
      dist = []
      for d in sorted_degrees:
        p = self._raw_data[b][d]["p"]
        if not p:
          dist.append("\\timeout{}")

        else:
          _ts = []
          if "Succeed" in self._raw_data[b][d]:
            _ts.extend(self._raw_data[b][d]["Succeed"])
          if "Failed" in self._raw_data[b][d]:
            _ts.extend(self._raw_data[b][d]["Failed"])
          _dist = [ t / (100*p) for t in _ts ]
          m, siqr = calc_siqr(_dist)
          _m_siqr = "\\mso{{{}}}{{{}}}{{}}".format(formatter(m), formatter(siqr))
          dist.append(_m_siqr)

      _tex = " & ".join(dist)
      self.log(" & {} \\\\".format(_tex))

    else:
      conds = []
      conds.append(PerfDB.match_EID(eid))
      conds.append(PerfDB.match_RID(e_name, pr_name))
      conds.append(PerfDB.match(pr_name, "BENCHMARK", b))
      strategies = self.__get_distinct("STRATEGY", [e_name, pr_name], conds)
      for s in split(strategies):
        _conds = conds[:]
        _conds.append(PerfDB.match(pr_name, "STRATEGY", s))
        cores = self.__get_distinct("CORE", [e_name, pr_name], _conds)
        if not cores: return
        self._raw_data[b][s] = {}
        for c in split(cores):
          self._raw_data[b][s][c] = {}
          self._stat_benchmark_parallel(eid, b, s, c)

      ## tex
      for s in self._raw_data[b]:
        for c in self._raw_data[b][s]:
          for col in self._raw_data[b][s][c]:
            _dist = self._raw_data[b][s][c][col]
            if "TIME" in col:
              _dist = [ t / 1000 for t in _dist ]
            m, siqr = calc_siqr(_dist)
            _m_siqr = "\\mso{{{}}}{{{}}}{{}}".format(formatter(m), formatter(siqr))
            self.log(" & ".join([s, str(c), col, _m_siqr]))

    if not single or not detailed: return

    # Wilcoxon test
    ss = 40 # sample size
    for d1, d2 in combinations(sorted_degrees, 2):
      # to estimate success rate, both successful and failed cases exist
      if "Succeed" not in self._raw_data[b][d1]: continue
      if "Succeed" not in self._raw_data[b][d2]: continue
      if "Failed" not in self._raw_data[b][d1]: continue
      if "Failed" not in self._raw_data[b][d2]: continue

      s1 = len(self._raw_data[b][d1]["Succeed"])
      f1 = len(self._raw_data[b][d1]["Failed"])
      if f1 < ss: continue # too few failure cases
      p1 = float(s1) / f1
      s2 = len(self._raw_data[b][d2]["Succeed"])
      f2 = len(self._raw_data[b][d2]["Failed"])
      if f2 < ss: continue # too few failure cases
      p2 = float(s2) / f2

      rs = []
      ps = []
      for i in xrange(100):
        ts1 = random.sample(self._raw_data[b][d1]["Failed"], ss)
        ts2 = random.sample(self._raw_data[b][d2]["Failed"], ss)
        dist_d1 = [ t1 / p1 for t1 in ts1 ]
        dist_d2 = [ t2 / p2 for t2 in ts2 ]
        rank_sum, pvalue = wilcoxon(dist_d1, dist_d2)
        rs.append(rank_sum)
        ps.append(pvalue)

      self.log("\nWilcoxon test for {} and {}".format(d1, d2))
      rs_percentile = calc_percentile(rs)
      s_rs_p = " | ".join(map(str, rs_percentile))
      self.log("rank: {} ({}) [ {} ]".format(np.mean(rs), np.var(rs), s_rs_p))
      ps_percentile = calc_percentile(ps)
      s_ps_p = " | ".join(map(str, ps_percentile))
      self.log("pvalue: {} ({}) [ {} ]".format(np.mean(ps), np.var(ps), s_ps_p))


  # statistics per experiment
  def _stat_exp(self, single, eid, detailed=False):
    r_name = sr_name if single else pr_name
    conds = []
    conds.append(PerfDB.match_EID(eid))
    conds.append(PerfDB.match_RID(e_name, r_name))
    benchmarks = self.__get_distinct("BENCHMARK", [e_name, r_name], conds)
    if not benchmarks:
      self.log("no benchmark for eid={}".format(eid))
      return
    for b in split(benchmarks):
      self._raw_data[b] = {}
      self._stat_benchmark(single, eid, b, detailed)


  # statistics
  def calc_stat(self, single=True, eid=0, detailed=False):
    if eid > 0:
      self._stat_exp(single, eid, detailed)
    else: # for all EIDs
      eids = self.__get_distinct("EID", [e_name], ["TRUE"])
      for (_eid,) in eids:
        self._stat_exp(single, _eid, detailed)


def main():
  parser = OptionParser(usage="usage: %prog [options]")
  parser.add_option("-c", "--cmd",
    action="store", dest="cmd",
    type="choice", choices=["init", "reset", "clean", "register", "stat"],
    default=None, help="command to run")
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
  parser.add_option("-f", "--file",
    action="append", dest="outputs", default=[],
    help="output files to post-analyze")
  parser.add_option("-s", "--single",
    action="store_true", dest="single", default=False,
    help="refer to backend behavior from single threaded executions")
  parser.add_option("--detail",
    action="store_true", dest="detail", default=False,
    help="calculate detailed statistics")
  parser.add_option("--dry",
    action="store", dest="dry", default=False,
    help="just print out insertion queries, not commit")
  parser.add_option("-v", "--verbose",
    action="store_true", dest="verbose", default=False,
    help="print every query")

  (opt, args) = parser.parse_args()

  if not opt.cmd:
    parser.error("nothing to do")

  global verbose
  verbose = opt.verbose

  db = PerfDB(opt.user, opt.db)

  if opt.cmd == "init":
    for t in schemas:
      if db.table_exists(t):
        print "The table already exists; nothing happened."
      else:
        db.init_table(t, schemas[t])

  elif opt.cmd == "reset":
    for t in schemas:
      if db.table_exists(t):
        db.drop_table(t)
      db.init_table(t, schemas[t])

  elif opt.cmd == "clean":
    for t in schemas:
      if db.table_exists(t):
        db.drop_table(t)

  elif opt.cmd == "register":
    for t in schemas:
      if not db.table_exists(t):
        raise Exception("{} not initialized".format(t))

    # if not specified, interpret all outputs under data/ folder
    if not opt.outputs:
      outputs = glob.glob(os.path.join(opt.data_dir, "*.txt"))
      # filter out erroneous cases (due to broken pipes, etc.)
      opt.outputs = filter(lambda f: os.path.getsize(f) > 0, outputs)

    db.reg_exp(opt.outputs, opt.single, opt.eid, opt.dry)

  elif opt.cmd == "stat":
    db.chk_integrity()
    db.calc_stat(opt.single, opt.eid, opt.detail)

  return 0


if __name__ == "__main__":
  sys.exit(main())

