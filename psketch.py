#!/usr/bin/env python

import datetime
import math
import multiprocessing
import os
import shutil
import subprocess32 as subp
import sys

DATA = "data"
b = ""
trials = 1
timeout = None
register = None


def find_arg(argv, arg, offset=0):
  for i, _arg in enumerate(argv):
    if arg in _arg:
      if arg == _arg: return argv[i+offset]
      elif "=" in _arg: return _arg.split("=")[-1]
  return None


def remove_arg(argv, arg):
  _argv = []
  for i in xrange(len(argv)):
    _arg = argv[i]
    if arg in _arg: continue
    _argv.append(_arg)
  return _argv


def repl_output_path(argv, path):
  _argv = argv[:]
  for i, arg in enumerate(argv):
    if arg == "-o": _argv[i+1] = path
  return _argv


def run(cmd, argv, seed):
  _timeout = timeout * 60 if timeout > 0 else None
  output_path = find_arg(argv, "-o", 1)
  _argv = argv[:]
  _argv.insert(0, str(seed))
  _argv.insert(0, "--seed")
  s_cmd = " ".join([cmd] + _argv)

  degree = find_arg(argv, "-randdegree", 1)
  output = os.path.join(DATA, "{}_single_{}_{}.txt".format(b, degree, str(seed)))
  res = False
  with open(output, 'w') as f:
    f.write("[psketch] {}{}".format(s_cmd, os.linesep))
    exit_code = -1
    try:
      exit_code = subp.check_call([cmd] + _argv, stdout=f, timeout=_timeout)
      if exit_code == 0: res = True
    except subp.CalledProcessError:
      f.write("[psketch] maybe failed{}".format(os.linesep))
    except subp.TimeoutExpired:
      f.write("[psketch] timed out: {}{}".format(_timeout*1000, os.linesep))
    f.write("[psketch] backend exit code: {}{}".format(exit_code, os.linesep))

  if register and register == "True":
    try:
      _opts = []
      _opts.extend(["-c", "register"])
      _opts.extend(["-f", output])
      _opts.extend(["-s"]) # single-threaded
      _opts.extend(["-e", "11"])
      #_opts.extend(["-v"])
      subp.check_call(["./db.py"] + _opts)
    except subp.CalledProcessError:
      print "database registration failed", output
    finally:
      if os.path.exists(output):
        os.remove(output)

  return (output_path, res)


def p_run(cmd, argv):
  output_path = find_arg(argv, "-o", 1)

  n_cpu = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(max(1, int(n_cpu * 0.83)))
  now = int(datetime.datetime.now().strftime("%H%M%S"))
  seed = now * (10 ** int(math.log(trials, 10)))

  def found( (fname, r) ):
    if r: # found, copy that output file
      shutil.copyfile(fname, output_path)
      #pool.close()
      #pool.terminate() # other running processes will become zombies here

  results = []
  temps = []
  try:
    for i in xrange(trials):
      _output_path = output_path + str(i)
      temps.append(_output_path)
      _argv = repl_output_path(argv, _output_path)
      r = pool.apply_async(run, (cmd, _argv, abs(seed+i)), callback=found)
      results.append(r)
    pool.close()
  except KeyboardInterrupt:
    pool.close()
    pool.terminate()
  except AssertionError: # apply_async is called after pool was terminated
    pass
  finally:
    pool.join()

  # clean up temporary files, while merging synthesis result
  res = False
  for i, fname in enumerate(temps):
    try:
      _fname, r = results[i].get(timeout=1) # very short timeout to kill zombies
      res = res or r
      assert fname == _fname
    except IndexError:
      pass # in case where temps.append happens but the loop finishes just before pool.apply_async
    except multiprocessing.TimeoutError: # zombie case
      pass
    finally:
      if os.path.exists(fname):
        os.remove(fname)

  return (output_path, res)


if __name__ == "__main__":
  sketch_home = os.environ["SKETCH_HOME"]
  if "runtime" in sketch_home: # using tar ball
    sketch_root = os.path.join(sketch_home, "..", "..")
  else: # from source
    sketch_root = os.path.join(sketch_home, "..")
  cegis = os.path.join(sketch_root, "sketch-backend", "src", "SketchSolver", "cegis")

  argv = sys.argv[1:]

  b = find_arg(argv, "-conc-benchmark")
  trials = int(find_arg(argv, "-conc-repeat"))
  timeout = int(find_arg(argv, "-conc-timeout"))
  register = find_arg(argv, "-conc-register")
  argv = remove_arg(argv, "-conc-benchmark")
  argv = remove_arg(argv, "-conc-repeat")
  argv = remove_arg(argv, "-conc-timeout")
  argv = remove_arg(argv, "-conc-register")
  _, res = p_run(cegis, argv)

  if res: sys.exit(0)
  else: sys.exit(1)

