"""
dvr_utils.py

A set of convenience methods for computing observables with DVR states.

Right now, this is empty. Most of the observable computations are done in the DVR class in dvr.py. This is 
nice because every instance of the class has access to the DVR mesh, n, L, etc., which many observables depend on.
However, I wonder if it'd be cleaner to leave DVR as a class strictly responsible for the DVR mesh, the Hamiltonian,
and the eigenvectors, and have another class (or a utils module) dedicated to computing observables.
"""