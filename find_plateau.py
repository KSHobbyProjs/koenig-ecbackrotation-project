#!/usr/bin/env python
"""
Find the optimal (n, L, theta training region) to train over.

Sweeps over values of the mesh width and L, selecting the pair
that minimizes the finite difference of a set of observables over theta.

With (n, L) fixed, sweeps over real theta, selecting a region in which
the observables vary the least. After this is found, sweeps over imaginary
theta, selecting a region in which observables vary the least over the 
resulting complex grid. The mesh width is nudged, and if the plateau remains
stable, this grid passes.

The code is all there, but it's slow when run in Jupyter. Plan to move it here.
"""
