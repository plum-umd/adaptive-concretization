# Adaptive Concretization for Parallel Program Synthesis

This project includes the testing infrastructure
to empirically evaluate our adaptive concretization algorithm
for parallel program synthesis.  For more details about
the algorithm and the usage of this infrastructure,
visit our GitHub page [here][gh].


## Structure

* \_\_init\_\_.py : a placeholder to enable other python modules to use each other
* benchmark/
* clean.sh : script to clean up remnants
* config.full.json : full version of experiments
* config.json : short version of experiments
* data/ : a place for Sketch outputs, figures, etc.
* db.py : script to manipulate database
* figure.py : script to draw figures
* post.py : script to post-analyze Sketch output
* psketch.py : script to run Sketch back-end in parallel
* run.py : script to run experiments
* simulate.py : script to simulate Wilcoxon test using database


## Usage

This section explains script usages in the order of experiments conducted.
For more details, again, refer to our GitHub page [here][gh].

#### run.py

This script is the main runner that runs Sketch in various settings:
* running plain Sketch,
* running only sketch-backend in parallel,
* running Sketch with fixed randdegree (see our paper for the terms), and
* running Sketch with adaptive concretization along with various numbers of cores.

All Sketch outputs will be saved under ```data/``` folder.

```sh
$ ./run.py -r #n [-b benchmark_name]* [-d degree]* [-c #n] [--strategy WILCOXON] [-s] [--timeout t]
```
You can run multiple benchmarks and
multiple degrees (if not using adaptive concretization):
```sh
$ ./run.py -r 3 -b colorchooser_demo -b reverse -d 16 -d 64 [-c 32]
```
To use adaptive concretization, specify the strategy as follows:
```sh
$ ./run.py [...] --strategy WILCOXON
```
To show our hypothesis that optimal randdegrees vary from benchmark to benchmark,
we run Sketch with several fixed randdegree many times.  To easily conduct
these experiments, this script has a feature to run only back-end in parallel:
```sh
$ ./run.py -s [...]
```
If you're interested, see `psketch.py` which acts as a wrapper for
Sketch's back-end script, `cegis`.

By default, `run.py` reads `config.json`, which is a subset of
`config.full.json` in terms of benchmarks, cores, and degrees configurations.
If you want to examine all benchmarks with all possible combinations of
configurations, pass `config.full.json` to `run.py`:
```sh
$ ./run.py --config config.full.json [...]
```

#### post.py

This script interprets Sketch outputs (under `data/` folder unless specifed elsewhere)
and retrives statistical information, such as elapsed time, number of trials,
chosen randdegree, etc.
```sh
$ ./post.py [-s] [-d data_dir] [-b benchmark_name]*
```

#### db.py

This script has various features to manipulate database:
* `-c init`: initializing tables
* `-c clean`: cleaning data in the tables
* `-c reset`: resetting tables
* `-c register`: registering statistics data (from `post.py`)
* `-c stat`: retriveing statistics

By default, this script assumes to access to the database
with username `sketchperf` and database name `concretization`.
(See GitHub page [here][gh] for how to set up database.)
If you are using different names, specify them:
```sh
$ ./db.py --user user_name --db db_name [...]
```
This script will read Sketch output files under `data/` folder,
but you can also specify either folder or single file:
```sh
$ ./db.py -d data_dir [...]
$ ./db.py -f path/to/file [...]
```
For the experiment about back-end behaviors, similar to `run.py`,
explicitly say it:
```sh
$ ./db.py -s [...]
```
If you want to see what queries this script makes, turn on the verbosity option:
```sh
$ ./db.py -v [...]
```

#### simulate.py

to be described later...

#### figure.py

to be decribed later...

[gh]: http://plum-umd.github.io/adaptive-concretization/

