# Adaptive Concretization for Parallel Program Synthesis

This project includes the testing infrastructure
to empirically evaluate our adaptive concretization algorithm
for parallel program synthesis.  For more details about
the algorithm and the usage of this infrastructure,
visit our github page [here][gh].

[gh]: http://plum-umd.github.io/adaptive-concretization/


## Structure

* \_\_init\_\_.py : a placeholder to enable other python modules to use each other
* benchmark/
* clean.sh : script to clean up remnants
* configure.full.json : full version of experiments
* configure.json : short version of experiments
* data/ : a place for Sketch outputs, figures, etc.
* db.py : script to manipulate database
* figure.py : script to draw figures
* post.py : script to post-analyze Sketch output
* psketch.py : script to run Sketch back-end in parallel
* run.py : script to run experiments
* simulate.py : script to simulate Wilcoxon test using database


## Usage

This section explains script usages in the order of experiments conducted.

#### run.py

#### post.py

#### db.py

#### simulate.py

#### figure.py

