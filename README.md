### MITgcm Parallel Analysis

This package is designed to make it possible to analyze very large MITgcm grids in parallel.

## Design Principles
These are the main ideas guiding the development.
1. The domain should be partitioned into manageable tiles; at no point should the whole domain be loaded into the memory of a single process (like the GCM execution itself).
1. But unlike the GCM execution, most analysis tasks are *embarassingly parallel*; communication between tiles is not required.
1. This means that our analysis tasks can be implemented using a [MapReduce](http://en.wikipedia.org/wiki/MapReduce) programming model.

## Implementation
The basic framework is the powerful [NumPy/SciPy stack](http://www.scipy.org/). In particular, NumPy's [memmap](http://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html) class allows us access small segments of large files on disk, without reading the entire file into memory. This is exactly what we need on each tile.

The parallelization is handled through the [IPython Parallel framework](http://ipython.org/ipython-doc/rel-1.1.0/parallel/index.html). This extremely flexible architecture makes it trivial to distribute execution in a wide range of environments, including MPI.

One of the biggest barriers against adopting python for scientific computing (over Matlab) is the expectation that it will be difficult to install. Forunately this barrier has been essentially eliminated by the recent emergence of completely pre-cooked NumPy/SciPy environments that are free for academic use. The two I have tested are:
* [Anaconda](https://store.continuum.io/cshop/anaconda/) by Continuum Analytics. This installed easily on pleaides in my home directory; hopefully it will be added as a module soon
* [Canopy](https://www.enthought.com/products/canopy/) by Enthought.




