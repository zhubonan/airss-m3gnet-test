# Testing M3GNet for Random Structure Searching

This repository contains results of tests for using M3GNet for random structure searching.

M3GNet is a Graph Neutron Network machine learning potential (MLP) that is trained on Materials Project database.
An important advance of this model is the use of trajectories of geometry optimisation in the training,
this improves the performance as well as the robustness of the model, making it one step closer to a universal MLP that covers the entire periodic table.

First-principle crystal structure prediction is often considered by very costly due to the need for DFT relaxations.
If M3GNet can completely replace the DFT calculation, this could result in significant savings of the computational cost.

In particular, we revist previous searches for battery materials $\ce{LiFePO4}$ and $\ce{LiFeSO4F}$, which have been conducted using *ab initio* random structue searching ([AIRSS](https://www.mtg.msm.cam.ac.uk/Codes/AIRSS)).
The searches has been conducted using [CASTEP](http://www.castep.org/) with the *QC5* ultrasoft pseudopotentials.
Further geometry optimisations are performed with more stringent pseudopotentials and setting for converged results, but often the initial results from the search can already give reasonable outputs.

Due to the stoichastic nature of the search, one cannot, in theory, fully *reproduce* the DFT results.
We try our best by using M3GNet to relax the *initial unrelaxed structures* from the DFT search.
Furthermore, the *relaxed DFT structures* may also be re-relaxed by M3GNet.

This tests are aimed at answersing two questions: first, whether M3GNet can reproduce the ordering of the low energy structures; second, whether M3GNet has a smooth potential energy surface (PES) which resemsbles that of the DFT.
While the former is needed to get the correction predicton, the latter is crucial for the MLP to be able find the low energy structure in the first place. 

Three test cases are included here, in the order of increasing chanllege: $\ce{LiFePO4}$, $\ce{LiFeSO4F}$, and $\ce{LiN2}$.


```{tableofcontents}
```
