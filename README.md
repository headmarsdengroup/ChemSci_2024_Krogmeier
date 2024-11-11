This repository contains the necessary scripts and electronic structure output data to reproduce the data described in, "Low temperature decoherence dynamics in molecular spin systems using the Lindblad master equation," T.J. Krogmeier, A.W. Schlimgen, and K. Head-Marsden, Chemical Science, 2024.

main.py contains the python class that will perform the ME-CCE 2nd order computation of spin dephasing due to the flip-flop of nuclear magnetization.

The directory dependencies contains all the code that main.py calls. It must be in the same directory as main.py.

The directory ORCA_output contains all of the electronic structure output. 
In it, there must be:

- a sub directory containing the electronic structure for geometries in an ensemble, including hyperfine tensor calculations.
- a .xyz file containing the equilibrium geometry for the molecule of interest.
- a .log file containing the output from ORCA, including computed hyperfine tensors for the equilibrium geometry.

data_gen.py is an example script that uses main.py to compute spin dephasing for a series of Vanadium and Copper containing molecular spin systems.

# TODO citation when paper details are finalized.
