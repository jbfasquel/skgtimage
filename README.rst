.. -*- mode: rst -*-

Overview
========

Skgtimage or scikit-gtimage (standing for ''graph tools for image analysis'') is a Python module for using graph in order to guide sequential image analysis, on top of numpy and networkx and distributed under the 3-Clause BSD license.

The important links are:

* Official source code repo: XXX
* HTML documentation : XXX

Dependencies
============

scikit-gtimage is tested to work under Python 2.7.

The required dependencies are:

* NumPy (>=1.8.0)
* SciPy (>=0.12.0)
* networkx (>=1.10)

For running the examples, you need Matplotlib >= 1.1.1.

For running the tests, you need nose >= 1.3.0.

For using the io subpackage, you need to install (checked with):

* pygraphviz (>=1.3.1), using pip for instance (pip install pygraphviz)
* pydot (>= 1.0.29), using pip for instance (pip install pydot2)
* PIL (>=3.0.0),  using pip for instance (pip install pillow)

For using the utils subpackage, you need to install (checked with):

* scikit-learn (>=0.17), using pip for instance (pip install sklearn)

Install
=======

The code can be retrieved as an archive from bitbucket or using git with the command::

    git clone https://xxxx/scikit-learn.git

The package can be installed using distutils ::

  python setup.py install

Note that an error may raised if scipy is not already installed. In such a case, install scipy manually (e.g. "pip install scipy") and rerun the scikit-gtimage installation.

The package and the installation can be tested using nose (assuming ``nose`` package installed)::

   nosetests -v skgtimage
