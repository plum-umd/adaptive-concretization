# Adaptive Concretization for Parallel Program Synthesis

This project includes the testing infrastructure
to empirically evaluate our adaptive concretization algorithm
for parallel program synthesis.  For more details about
the algorithm and the usage of this infrastructure,
visit our GitHub page [here][gh].


## Publication

* [Adaptive Concretization for Parallel Program Synthesis][cav].
  Jinseong Jeon, Xiaokang Qiu, Armando Solar-Lezama, and Jeffrey S. Foster.
  In International Conference on Computer Aided Verification (CAV '15), Jul 2015.


[cav]: http://dx.doi.org/10.1007/978-3-319-21668-3_22


## Structure

* \_\_init\_\_.py : a placeholder to enable other python modules to use each other
* benchmark/
* clean.sh : script to clean up remnants
* config.full.json : full version of experiments
* config.short.json : short version of experiments
* data/ : a place for Sketch outputs, figures, etc.
* db.py : script to manipulate database
* figure.py : script to draw figures
* post.py : script to post-analyze Sketch output
* psketch.py : script to run Sketch back-end in parallel
* run.py : script to run experiments
* simulate.py : script to simulate online strategies using database
* util.py : utilities
* wilcoxon.py : script to simulate Wilcoxon test using Monte Carlo method


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
$ ./run.py -r n [-b benchmark_name]* [-d degree]* [-c n] [--strategy WILCOXON] [-s] [--timeout t]
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

By default, `run.py` reads `config.short.json`, which is a subset of
`config.full.json` in terms of benchmarks, cores, and degrees configurations.
If you want to examine all benchmarks with all possible combinations of
configurations, pass `config.full.json` to `run.py`:
```sh
$ ./run.py --config config.full.json [...]
```


#### post.py

This script interprets Sketch outputs (under `data/` folder unless specifed elsewhere)
and retrieves statistical information, such as elapsed time, number of trials,
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
* `-c stat`: retrieving statistics

By default, this script assumes to access to the database
with username `sketchperf` and database name `concretization`.
(See our GitHub page [here][db] for how to set up database.)
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
explicitly specify it (along with a distinct EID number):
```sh
$ ./db.py -s -e 11 [...]
```
If you want to see what queries this script makes, turn on the verbosity option:
```sh
$ ./db.py -v [...]
```


#### simulate.py

As a proof-of-concept, this script simulates several
online concretization strategies, such as random, fixed degree,
minimal time, and Wilcoxon-based statistical comparison.
The calculation of expected running time is based on the empirical
data in the database, especially the experiment about back-end behaviors,
where SAT propagation information is recorded.
If you use a different EID for that experiment
(see our GitHub page [here][gh] and/or `db.py` above),
you may need to specify it.
```sh
$ ./simulate.py [-e EID]
```
Since the simulation depends on the empirical data in the database,
similar options for the database are available:
```sh
$ ./simulate.py [--user user_name] [--db db_name]
```

The output of this script is simulated running time,
along with variance and quantiles.  I.e., the smaller, the better.
However, Note that it is quite tricky to simulate parallel running,
so this simulation result can not guarantee the best solution.


#### wilcoxon.py

To set a good confidential cutoff (so-called p-value) and
the number of samplings, this script simulates Wilcoxon test
by using Monte Carlo method: given configuration of the number
of sampling and two target degrees, it will randomly pick
that number of samples from the database; compare the given degrees
using that sampling; and repeat this process many times.
The distributions of two degrees are computed as
the multiplication of running time and search space,
which could be retrieved from backend data set.

Similar to `simulate.py`, you may need to specify your own EID if any.
```sh
$ ./wilcoxon.py [-e EID]
```
This script uses the same backend data as `simulate.py` does,
and similar options for the database are available:
```sh
$ ./wilcoxon.py [--user user_name] [--db db_name]
```


#### figure.py

Although tables with detailed numbers are comprehensive,
those are often too verbose and not really useful for presenting
the meanings in the results.  This script is designed to convert
raw data generated by `db.py` to *graph*s.

For the experiment about back-end behaviors, similar to `run.py`
and `db.py` above, explicitly specify it (along with a distinct
EID number):
```sh
$ ./figure.py -s -e 11 [...]
```
It will generate figures _per benchmark_;
x-axis of those figures is various degrees, whereas
y-axis is either average size of SAT formulae propagation or
average time of successful cases and failure cases.
Besides, it will also generate so-called vee chart that shows
the idea that optimal degrees vary from benchmark to benchmark.

For the experiments about adaptive concretization and its
scalability, just run the script:
```sh
$ ./figure.py [...]
```
It will generate two figures: the one that compares adaptive
concretization against the plain Sketch, and the other that
contrasts adaptive concretization with different numbers of cores.


[gh]: http://plum-umd.github.io/adaptive-concretization/
[db]: http://plum-umd.github.io/adaptive-concretization/exp.html#DB

