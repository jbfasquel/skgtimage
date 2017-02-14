.. -*- mode: rst -*-

Overview
========

The scikit-gtimage (standing for ''graph tools for image analysis''), skgtimage for package aims at providing graph-based tools for image interpretation.
It implements a method, based on an inexact graph matching technique, for retrieving and identifying a set image regions, from an initial oversegmentation,
using a priori declared qualitative inclusion and photometric relationships between the set of image regions.
This package is distributed under the 3-Clause BSD license.

Installation
============

Dependencies
************

The proposed scikit-gtimage package is tested to work under Python 3.5.2 (should work for > 3.5.2), not with Python 2 !

The required dependencies are:

* numpy (tested with 1.11.3)
* scipy (tested with 0.18.1)
* Pillow (tested with 4.0.0)
* matplotlib (tested with 2.0.0)
* networkx (tested with 1.11)
* scikit-image (tested with 0.12.3)
* scikit-learn (tested with 0.18.1)

Using miniconda, required dependencies can be installed as following:

* install miniconda (python 3.5): https://conda.io/miniconda.html
* conda install numpy
* conda install scipy
* conda install Pillow
* conda install matplotlib
* conda install networkx
* conda install scikit-image
* conda install scikit-learn

Optionally (not required), for displaying and exporting graphs, you need to install pygraphviz (using pip for instance: pip install pygraphviz).
Note that pygraphviz requires graphviz.

Testing
*******

Simply unzip the archive and run provided examples (from the root directory of the unzipped archive) :

python example1.py

python example2.py

Note: depending on your system, you may run: python3 example1.py , python3 example2.py

Using
*****

Add the root directory of the archive to the PYTHONPATH environment variable.

