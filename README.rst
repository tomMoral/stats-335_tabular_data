Benchmark for classification methods on tabular data
====================================================
|Build Status| |Python 3.8+|


Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to **tabular classification and regression methods**.

The objective is to compare the performance of different ML algorithms on
various tabular datasets. The performance are evaluated on the test set,
to evaluate the generalization performance of the algorithms.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/tomMoral/stats-335_tabular_data
   $ cd stats-335_tabular_data
   $ benchopt run

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run -s solver1 -d dataset2 --max-runs 10 --n-repetitions 1


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/tomMoral/stats-335_tabular_data/workflows/Tests/badge.svg
   :target: https://github.com/tomMoral/stats-335_tabular_data/actions
.. |Python 3.8+| image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/release/python-380/
