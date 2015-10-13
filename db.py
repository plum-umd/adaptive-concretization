#!/usr/bin/env python

import cStringIO
from itertools import combinations
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

import util
import post

"""
table Experiment
+-----------+
| EID | RID |
+___________+

table RunS (key: auto_inc RID)
+------------------------------------------------------------------------+
| RID | Benchmark | Degree | Succeed | Seed | Propagation | Space | Time |
+------------------------------------------------------------------------+

table Dag (key: RID + Harness)
+----------------------+
| RID | Harness | Size |
+----------------------+

table Hole (key: auto_inc HID)
+-------------------------------------------+
| RID | HID | Replaced | Name | Odds | Size |
+-------------------------------------------+

table RunP (key: auto_inc RID)
+----------------------------------------------------------------------------------------------+
| RID | Benchmark | Strategy | Core | Degree | Succeed | Trial | TTime | STime | FTime | CTime |
+----------------------------------------------------------------------------------------------+

"""

# constants
t_b = "BIT"
t_i = "INT"
t_f = "FLOAT"
t_d = "DOUBLE"
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
  ["SPACE", t_d, nnull],
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
  ["FTIME", t_f, nnull],
  ["CTIME", t_i, nnull]
]

schemas = {
  e_name: e_schema,
  sr_name: sr_schema,
  d_name: d_schema,
  h_name: h_schema,
  pr_name: pr_schema
}

verbose = False

class PerfDB(object):

  def __init__(self, user="sketchperf", db="concretization"):
    self.cnx = mysql.connector.connect(host="127.0.0.1", user=user)
    self.cnx.database = db # assume this db is already set up
    self.cnx.get_warnings = True
    self._drawing = False
    self._detail_space = False
    self._detail_full = False
    self._raw_data = {}

  @property
  def drawing(self):
    return self._drawing

  @drawing.setter
  def drawing(self, v):
    self._drawing = v

  @property
  def detail_space(self):
    return self._detail_space

  @detail_space.setter
  def detail_space(self, v):
    self._detail_space = v

  @property
  def detail_full(self):
    return self._detail_full

  @detail_full.setter
  def detail_full(self, v):
    self._detail_full = v

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
      FROM information_schema.TABLES
      WHERE TABLE_SCHEMA = '{0}' and TABLE_NAME = '{1}'
    """.format(self.cnx.database, util.sanitize_table_name(table_name))
    PerfDB.__execute(cur, query)
    ret = False
    if cur.fetchone()[0] == 1: ret = True
    cur.close()
    return ret

  # True if the column of interest indeed exists
  def col_exists(self, table_name, col_name):
    cur = self.cnx.cursor()
    query = """
      SELECT COUNT(*)
      FROM information_schema.COLUMNS
      WHERE TABLE_SCHEMA = '{0}' and TABLE_NAME = '{1}'
          and COLUMN_NAME = '{2}'
    """.format(self.cnx.database, util.sanitize_table_name(table_name), col_name)
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
    """.format(util.sanitize_table_name(table_name), s_schema)
    PerfDB.__execute(cur, query)
    cur.close()

  # delete the table
  def drop_table(self, table_name):
    cur = self.cnx.cursor()
    query = """
      DROP TABLE IF EXISTS {0}
    """.format(util.sanitize_table_name(table_name))
    PerfDB.__execute(cur, query)
    cur.close()

  insert_query = """
    INSERT INTO {0} ({1}) VALUES
      {2}
  """

  # register one single-threaded backend record
  def __reg_record_single(self, output, record):
    # register "RunS"
    _record = {}
    for k in record:
      if type(record[k]) in [list, dict]: continue
      _record[k] = record[k]

    cur = self.cnx.cursor()
    up_call = op.methodcaller("upper")

    up_keys = ','.join(map(up_call, _record.keys()))
    _values = '(' + ", ".join(map(util.quote, _record.values())) + ')'
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
        return '(' + ','.join(map(util.quote, [rid] + dic.values())) + ')'
      _values = map(to_val, lst)
      valuess = ",\n\t".join(_values)
      PerfDB.__execute(cur, PerfDB.insert_query.format(table_name, up_keys, valuess))

    # register "Dag"
    if record["dag"]: reg_list_of_dict(d_name, record["dag"])

    # register "Hole"
    if record["hole"]: reg_list_of_dict(h_name, record["hole"])

    cur.close()
    return rid


  # register one single-threaded backend run
  def __reg_run_single(self, output):
    benchmark, strategy, core, degree = post.find_config(output, True)
    if not benchmark: return
    self.log("register: {}".format(output))
    record = post.be_analyze(output, benchmark, strategy, degree)
    return self.__reg_record_single(output, record)


  # register one parallel run
  def __reg_run_parallel(self, output):
    benchmark, strategy, core, degree = post.find_config(output, False)
    if not benchmark: return
    self.log("register: {}".format(output))
    record = post.analyze(output, benchmark, strategy, core, degree)

    # register "RunP"
    _record = {}
    be_rids = []
    for k in record:
      if type(record[k]) in [list, dict]:
        # register internal backend outputs
        if "backend" in k:
          for be_record in record[k]:
            _rid = self.__reg_record_single(output, be_record)
            be_rids.append(_rid)
        continue
      _record[k] = record[k]

    cur = self.cnx.cursor()
    up_call = op.methodcaller("upper")

    up_keys = ','.join(map(up_call, _record.keys()))
    _values = '(' + ", ".join(map(util.quote, _record.values())) + ')'
    PerfDB.__execute(cur, PerfDB.insert_query.format(pr_name, up_keys, _values))
    rid = cur.lastrowid
    if not rid:
      self.log("failed to register: {}".format(output))
      cur.close()
      return None

    cur.close()
    return rid, be_rids


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
    be_rids = []
    for output in outputs:
      if single:
        rid = self.__reg_run_single(output)
      else:
        rid, _be_rids = self.__reg_run_parallel(output)
        if _be_rids: be_rids.extend(_be_rids)
      if rid: rids.append(rid)

    def assoc_eid_rids(eid, rids):
      cur = self.cnx.cursor()
      paired_rids = map(lambda rid: "({},{})".format(eid, rid), rids)
      joined_rids = ", ".join(paired_rids)
      PerfDB.__execute(cur, PerfDB.insert_query.format(e_name, "EID, RID", joined_rids))
      cur.close()

    if rids: assoc_eid_rids(eid, rids)
    if be_rids: assoc_eid_rids(eid+10, be_rids)

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
    return "{}.{} = {}".format(table_name, what, util.quote(v))

  # statistics per benchmark and degree
  def _stat_benchmark_degree_single(self, eid, b, d):
    self.log("\nbenchmark: {}, degree: {}".format(b, d))

    conds = []
    conds.append(PerfDB.match_EID(eid))
    conds.append(PerfDB.match_RID(e_name, sr_name))
    conds.append(PerfDB.match(sr_name, "BENCHMARK", b))
    conds.append(PerfDB.match(sr_name, "DEGREE", d))
    n_total = self.__select_one(["COUNT"], ["*"], [e_name, sr_name], conds, None)[0]

    ## success rate
    _conds = conds[:]
    _conds.append(PerfDB.match(sr_name, "SUCCEED", "Succeed"))
    n_succeed = self.__select_one(["COUNT"], ["*"], [e_name, sr_name], _conds, None)[0]

    self.log("success rate: {} / {}".format(n_succeed, n_total))
    p = float(n_succeed) / n_total
    self._raw_data[b][d]["p"] = p

    ## (successful/failed) time
    def __stat_ttime(succeed):
      _conds = conds[:]
      _conds.append(PerfDB.match(sr_name, "SUCCEED", succeed))

      dist = self.__select_one(["MIN", "MAX", "AVG", "STDDEV"], ["TTIME"], [e_name, sr_name], _conds, None)
      _min, _max, _avg, _dev = dist
      _ttimes = self.__select_all([], ["TTIME"], [e_name, sr_name], _conds, None)
      _percentile = ""
      if _ttimes:
        self._raw_data[b][d][succeed] = util.split(_ttimes)
        _vals = util.calc_percentile(_ttimes)
        _percentile = "[" + " | ".join(map(str, _vals)) + "]"

      lucky = "lucky" if succeed == "Succeed" else "unlucky"
      self.log("{} runs' solution time: {} ({})".format(lucky, _avg, _dev))
      if _percentile: self.log("\tquantiles: {}".format(_percentile))

    __stat_ttime("Succeed")
    __stat_ttime("Failed")

    # estimated running time
    if not p:
      self._raw_data[b][d]["E(t)"] = (float("inf"), float("inf"))
    else:
      _ts = []
      if "Succeed" in self._raw_data[b][d]:
        _ts.extend(self._raw_data[b][d]["Succeed"])
      if "Failed" in self._raw_data[b][d]:
        _ts.extend(self._raw_data[b][d]["Failed"])
      _dist = [ t / (100*p) for t in _ts ]
      m, siqr = util.calc_siqr(_dist)
      self._raw_data[b][d]["E(t)"] = (m, siqr)
    _et, _et_siqr = self._raw_data[b][d]["E(t)"]
    self.log("estimated running time: {} ({})".format(_et, _et_siqr))

    if not self._detail_space and not self._detail_full: return

    ## search space

    # SPACE is newly added, so it may not exist in old db/dumps
    if self.col_exists(sr_name, "SPACE"):
      spaces = self.__select_all([], ["SPACE"], [e_name, sr_name], conds, None)

    else:
      _conds = conds[:]
      _conds.append(PerfDB.match_RID(sr_name, h_name))

      spaces = []
      _rids = self.__get_distinct(h_name+".RID", [e_name, sr_name, h_name], _conds)
      for (_rid,) in _rids:
        __conds = _conds[:]
        __conds.append(PerfDB.match(h_name, "RID", _rid))
        hole_szs = self.__select_all([], ["SIZE"], [e_name, sr_name, h_name], __conds, "NAME")
        _, _szs = util.split(hole_szs)
        space = float(reduce(op.mul, _szs))
        spaces.append(space)

    self._raw_data[b][d]["search space"] = util.split(spaces)
    _percentile = util.calc_percentile(spaces)
    self.log("search space: [{}]".format(" | ".join(map(str, _percentile))))

    if not self._detail_full: return

    ## propagation
    props = self.__select_all([], ["PROPAGATION"], [e_name, sr_name], conds, None)
    self._raw_data[b][d]["propagation"] = util.split(props)
    _percentile = util.calc_percentile(props)
    self.log("propagation: [{}]".format(" | ".join(map(str, _percentile))))

    ## DAG
    _conds = conds[:]
    _conds.append(PerfDB.match_RID(sr_name, d_name))
    dag = self.__select_all(["AVG"], ["SIZE"], [e_name, sr_name, d_name], _conds, "HARNESS")

    self.log("average dag size:")
    for seq in dag:
      hrns, avg_sz = seq
      self.log("  {}: {}".format(hrns, avg_sz))

    ## concretization rate
    _conds = conds[:]
    _conds.append(PerfDB.match_RID(sr_name, h_name))
    _conds.append(PerfDB.match(h_name, "REPLACED", "Replaced"))
    rpl = self.__select_one(["COUNT"], ["*"], [e_name, sr_name, h_name], _conds, None)[0]
    self.log("concretization rate: {}".format(float(rpl) / n_total))

    ## hole concretization histogram
    rpl_holes = self.__select_all(["COUNT"], ["*"], [e_name, sr_name, h_name], _conds, "NAME")
    hole_dict = util.to_dict(rpl_holes)
    s_hole_dict = {}
    if n_succeed:
      _conds.append(PerfDB.match(sr_name, "SUCCEED", "Succeed"))
      s_holes = self.__select_all(["COUNT"], ["*"], [e_name, sr_name, h_name], _conds, "NAME")
      s_hole_dict = util.to_dict(s_holes)

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
        self._raw_data[b][s][c][cname] = util.split(_cols)
        _vals = util.calc_percentile(_cols)
        _percentile = "[" + " | ".join(map(str, _vals)) + "]"

      self.log("{}: {} ({})".format(cname, _avg, _dev))
      if _percentile: self.log("\tquantiles: {}".format(_percentile))

    __stat_col("TRIAL")

    __stat_col("FTIME")
    __stat_col("STIME")
    __stat_col("TTIME")
    # CPU TIME is newly added, so it may not exist in old db/dumps
    if self.col_exists(pr_name, "CTIME"):
      __stat_col("CTIME")

    __stat_col("DEGREE")
    _conds = conds[:]
    degrees = self.__select_all(["COUNT"], ["*"], [e_name, pr_name], _conds, "DEGREE")
    d_dict = util.to_dict(degrees)
    s_d_dict = {}
    if n_succeed:
      _conds.append(PerfDB.match(pr_name, "SUCCEED", "Succeed"))
      s_degrees = self.__select_all(["COUNT"], ["*"], [e_name, pr_name], _conds, "DEGREE")
      s_d_dict = util.to_dict(s_degrees)

    self.log("degree histogram")
    for degree in sorted(d_dict.keys()):
      cnt = d_dict[degree]
      s_cnt = " ({})".format(s_d_dict[degree]) if degree in s_d_dict else ""
      self.log("  {}: {}{}".format(degree, cnt, s_cnt))


  def __get_distinct(self, col, tables, conds):
    return self.__select_all(["DISTINCT"], [col], tables, conds, None)


  # statistics per benchmark
  def _stat_benchmark(self, single, eid, b):
    if single:
      conds = []
      conds.append(PerfDB.match_EID(eid))
      conds.append(PerfDB.match_RID(e_name, sr_name))
      conds.append(PerfDB.match(sr_name, "BENCHMARK", b))
      degrees = self.__get_distinct("DEGREE", [e_name, sr_name], conds)
      if not degrees: return
      sorted_degrees = sorted(util.split(degrees))
      for d in sorted_degrees:
        util.init_k(self._raw_data[b], d)
        self._stat_benchmark_degree_single(eid, b, d)

      buf = cStringIO.StringIO()
      _d_hist = []
      for d in sorted_degrees:
        _hist = self._raw_data[b][d]
        s_cnt = len(_hist["Succeed"]) if "Succeed" in _hist else 0
        f_cnt = len(_hist["Failed"]) if "Failed" in _hist else 0
        _cnt = s_cnt + f_cnt
        buf.write("  {}: {}\n".format(d, _cnt))
        _d_hist.append(_cnt)
      s_q = " | ".join(map(str, util.calc_percentile(_d_hist)))
      self.log("{}degree histogram: [ {} ]".format(os.linesep, s_q))
      self.log(buf.getvalue())

      ## tex
      dist = []
      for d in sorted_degrees:
        p = self._raw_data[b][d]["p"]
        if not p:
          dist.append("\\timeout{}")

        else:
          m, siqr = self._raw_data[b][d]["E(t)"]
          _m_siqr = "\\mso{{{}}}{{{}}}{{}}".format(util.formatter(m), util.formatter(siqr))
          dist.append(_m_siqr)

      _tex = " & ".join(dist)
      self.log(" & {} \\\\".format(_tex))

    else:
      conds = []
      conds.append(PerfDB.match_EID(eid))
      conds.append(PerfDB.match_RID(e_name, pr_name))
      conds.append(PerfDB.match(pr_name, "BENCHMARK", b))
      strategies = self.__get_distinct("STRATEGY", [e_name, pr_name], conds)
      for s in util.split(strategies):
        _conds = conds[:]
        _conds.append(PerfDB.match(pr_name, "STRATEGY", s))
        cores = self.__get_distinct("CORE", [e_name, pr_name], _conds)
        if not cores: return
        util.init_k(self._raw_data[b], s)
        for c in util.split(cores):
          util.init_k(self._raw_data[b][s], c)
          self._stat_benchmark_parallel(eid, b, s, c)

      ## tex
      for s in self._raw_data[b]:
        for c in self._raw_data[b][s]:
          for col in self._raw_data[b][s][c]:
            _dist = self._raw_data[b][s][c][col]
            if "TIME" in col:
              _dist = [ t / 1000 for t in _dist ]
            m, siqr = util.calc_siqr(_dist)
            _m_siqr = "\\mso{{{}}}{{{}}}{{}}".format(util.formatter(m), util.formatter(siqr))
            self.log(" & ".join([s, str(c), col, _m_siqr]))

    if not single or not self._detail_full: return

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
      p1 = float(s1) / (s1 + f1)
      s2 = len(self._raw_data[b][d2]["Succeed"])
      f2 = len(self._raw_data[b][d2]["Failed"])
      if f2 < ss: continue # too few failure cases
      p2 = float(s2) / (s2 + f2)

      rs = []
      ps = []
      for i in xrange(301):
        ts1 = random.sample(self._raw_data[b][d1]["Failed"], ss)
        ts2 = random.sample(self._raw_data[b][d2]["Failed"], ss)
        dist_d1 = [ t1 / p1 for t1 in ts1 ]
        dist_d2 = [ t2 / p2 for t2 in ts2 ]
        rank_sum, pvalue = wilcoxon(dist_d1, dist_d2)
        rs.append(rank_sum)
        ps.append(pvalue)

      self.log("\nWilcoxon test for {} and {}".format(d1, d2))
      rs_percentile = util.calc_percentile(rs)
      s_rs_p = " | ".join(map(str, rs_percentile))
      self.log("rank: {} ({}) [ {} ]".format(np.mean(rs), np.var(rs), s_rs_p))
      ps_percentile = util.calc_percentile(ps)
      s_ps_p = " | ".join(map(str, ps_percentile))
      self.log("pvalue: {} ({}) [ {} ]".format(np.mean(ps), np.var(ps), s_ps_p))


  # statistics per experiment
  def _stat_exp(self, benchmarks, single, eid):
    r_name = sr_name if single else pr_name
    conds = []
    conds.append(PerfDB.match_EID(eid))
    conds.append(PerfDB.match_RID(e_name, r_name))

    if not benchmarks:
      _benchmarks = self.__get_distinct("BENCHMARK", [e_name, r_name], conds)
      if not _benchmarks:
        self.log("no benchmark for eid={}".format(eid))
        return
      benchmarks = util.split(_benchmarks)

    for b in benchmarks:
      util.init_k(self._raw_data, b)
      self._stat_benchmark(single, eid, b)


  # statistics
  def calc_stat(self, benchmarks, single=True, eid=0):
    if eid > 0:
      self._stat_exp(benchmarks, single, eid)
    else: # for all EIDs
      eids = self.__get_distinct("EID", [e_name], ["TRUE"])
      for (_eid,) in eids:
        self._stat_exp(benchmarks, single, _eid)


def main():
  parser = OptionParser(usage="usage: %prog [options] (output_file)*")
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
  parser.add_option("-b", "--benchmark",
    action="append", dest="benchmarks", default=[],
    help="benchmark(s) of interest")
  parser.add_option("-s", "--single",
    action="store_true", dest="single", default=False,
    help="refer to backend behavior from single threaded executions")
  parser.add_option("--detail-space",
    action="store_true", dest="detail_space", default=False,
    help="calculate detailed statistics, including search space")
  parser.add_option("--detail-full",
    action="store_true", dest="detail_full", default=False,
    help="calculate fully detailed statistics")
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
  db.detail_space = opt.detail_space
  db.detail_full = opt.detail_full

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
    if not args:
      outputs = glob.glob(os.path.join(opt.data_dir, "*.txt"))
      # filter out erroneous cases (due to broken pipes, etc.)
      outputs = filter(lambda f: os.path.getsize(f) > 0, outputs)
    else:
      outputs = args

    db.reg_exp(outputs, opt.single, opt.eid, opt.dry)

  elif opt.cmd == "stat":
    db.chk_integrity()
    db.calc_stat(opt.benchmarks, opt.single, opt.eid)

  return 0


if __name__ == "__main__":
  sys.exit(main())

