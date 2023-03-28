# Testing M3GNet for Random Structure Searching


This repository contains results of tests for using M3GNet for random structure searching.

The generated Jupyter Book website is hosted here 👉 https://zhubonan.github.io/airss-m3gnet-test/

M3GNet is a Graph Neutron Network machine learning potential (MLP) that is trained on Materials Project database.
An important advance of this model is the use of trajectories of geometry optimisation in the training,
this improves the performance as well as the robustness of the model, making it one step closer to a universal MLP that covers the entire periodic table.

First-principle crystal structure prediction is often considered by very costly due to the need for DFT relaxations.
If M3GNet can completely replace the DFT calculation, this could result in significant savings of the computational cost.

In particular, we revisit previous searches for battery materials $\ce{LiFePO4}$ and $\ce{LiFeSO4F}$, which have been conducted using *ab initio* random structure searching ([AIRSS](https://www.mtg.msm.cam.ac.uk/Codes/AIRSS)).
The searches have been conducted using [CASTEP](http://www.castep.org/) with the *QC5* ultrasoft pseudopotentials.
Further geometry optimisations are performed with more stringent pseudopotentials and setting for converged results, but often the initial results from the search can already give reasonable outputs.

Due to the stochastic nature of the search, one cannot, in theory, fully *reproduce* a search.
We try our best by using M3GNet to relax the *initial unrelaxed structures* from the DFT search.
Furthermore, the *relaxed DFT structures* may also be re-relaxed by M3GNet.

These tests are aimed at answering two questions: first, whether M3GNet can reproduce the ordering of the low energy structures; second, whether M3GNet has a smooth potential energy surface (PES) which resembles that of the DFT.
While the former is needed to get the correction prediction, the latter is crucial for the MLP to be able to find the low energy structure in the first place. 

Three test cases are included here, in the order of increasing challenge: $\ce{LiFePO4}$, $\ce{LiFeSO4F}$, and $\ce{LiN2}$.



## Relaxations using M3GNet

A `m3gnet-relax.py` script is avaliable for relaxing structures:


```
python m3gnet-relax.py <outdir> <file1> <file2> ....
```

The signature is designed in away to allow easy parallelised executions using [parallel](https://www.gnu.org/software/parallel/). 

For example, to relax the DFT-relaxed LiFePO4 structure using M3GNet with 8 processes in parallel with 30 structures in each batch:

```
ls with-u-rerun/*.res | parallel -j 8 -n 30 python m3gnet-relax.py with-u-rerun-relaxed-m3gnet {}
```


# Table of contents

:::{tableofcontents}
:::
