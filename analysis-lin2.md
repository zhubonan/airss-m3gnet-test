---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Test case - $\ce{LiN2}$

In this example, we search for $\ce{LiN2}$ using both DFT and M3GNet. 
The DFT search is performed first using AIRSS, the M3GNet is used to relax the initial structures and the DFT relaxed structures.

This compound, $\ce{LiN2}$ has not been reported experimentally, 
and there is no entry of such stoichiometry in the Materials Project database (as of March 2023).

```{code-cell} ipython3
:tags: [hide-input]
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

import warnings
from pathlib import Path
from ase.io import read
import ase
import pandas as pd

from tqdm import tqdm
tqdm = lambda x: x

def load_dataset(names):
    cells = [read(x) for x in  tqdm(names)]
    for atoms, name in zip(cells, input_names):
        atoms.info['fname'] = name.stem

    dataset = []
    for atoms in cells:
        dataset.append(
        {
            'atoms': atoms,
            'label': atoms.info['fname'],
            'energy': atoms.info['energy'],
            'energy_per_atom': atoms.info['energy'] / len(atoms),
            'volume': atoms.get_volume(),
            'volume_per_atom': atoms.get_volume() / len(atoms),

        }
        )

    return pd.DataFrame(dataset).sort_values('energy_per_atom')

def show_compact(df):
    return df[['label', 'energy_per_atom', 'volume_per_atom']]

def add_labels(ax):
    """Add axis labels"""
    ax.set_xlabel('Energy per atom (eV)')
    ax.set_ylabel(r'Volume per atom ($\mathrm{\AA^3}$)')


```

Load calculated data from files

```{code-cell} ipython3
:tags: [hide-input]
input_names = list(Path("lin2/run1/").glob("*.res"))
df_dft = load_dataset(input_names)

input_names = list(Path("lin2/run1-m3gnet/").glob("*.res"))
df_m3g = load_dataset(input_names)

input_names = list(Path("lin2/run1-relaxed-m3gnet").glob("*.res"))
df_m3g_from_relaxed = load_dataset(input_names)
```

## Energy - Volume distribution

Similar to the other two cases, we plot the energy per atom against the volume per atom.

Distribution from the M3GNet search is shown below.

```{code-cell} ipython3
:tags: [hide-input]
ax = df_m3g.plot.scatter('volume_per_atom', 'energy_per_atom', label='m3gnet')
ax = df_m3g_from_relaxed.plot.scatter('volume_per_atom', 'energy_per_atom',  xlim=(9, 20), label='m3gnet-from-dft', ax=ax, color='C1')
ax.set_ylim(df_m3g.energy_per_atom.min() - 0.01, df_m3g.energy_per_atom.min() + 0.1)
ax.legend(loc=1)
add_labels(ax)
```

Distribution from the DFT search, which is shown below, is quite different from that of the M3GNet.

```{code-cell} ipython3
:tags: [hide-input]
ax = df_dft.plot.scatter('volume_per_atom', 'energy_per_atom', 
                         xlim=(9, 20), ylim=(-434, -433.75),
                         label='PBE Search')
ax.set_ylim(df_dft.energy_per_atom.min() - 0.01, df_dft.energy_per_atom.min() + 0.1)
ax.legend(loc=1)
add_labels(ax)
```

In addition, the lowest energy structure have very different volumes, as the lowest
energy structures from the M3GNet are:
```{code-cell} ipython3
:tags: [hide-input]
show_compact(df_dft.iloc[:10])
```
and those from the DFT search are:

```{code-cell} ipython3
:tags: [hide-input]
show_compact(df_m3g.iloc[:10])
```

Checking the lowest energy structures visually, we found:
    
- M3GNet relaxed structures feature azide ions, while the DFT ones contains the N2 dimers.
- M3GNet relaxed structure are generally high in volume compared to the DFT results.


## Energy distribution of the relaxed structures

The distributions from the two searches are very different, as shown below:

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

engs = df_m3g['energy_per_atom'].values.copy()
engs_m3g = engs -engs.min()

engs = df_dft['energy_per_atom'].values.copy()
engs_dft = engs - engs.min()

plt.hist(engs_m3g, bins=100,alpha=0.5,density=True, label='M3GNet', range=(0,0.5));
plt.hist(engs_dft, bins=100, alpha=0.5, density=True, label='PBE + U', range=(0, 0.5))
plt.legend()
plt.xlabel('Energy per atom (eV)')
plt.ylabel('Probability density');
```
Clearly, this case the M3GNet does not reproduce the PES of the DFT at all, since the two distributions are vastly different.
It may appear that M3GNet has a high concentration of low energy structure, 
this only is due to the reference energy is taken from the lowest energy structure using the same methodology. 

## Which approach finds lower energy structure?

It is not possible to compare the energies directly as the DFT calculations are done with CASTEP+QC5 while the M3GNet 
is trained on VASP+PBE,
and unlike previous two cases, we cannot use known lowest structures as references for comparison.
The QC5 pseudopotentials used in the search are not designed to be highly transferable but focus on the speed instead.
Hence, we re-relax the structures generated from both approaches using CASTEP+C19 at 800 eV and a kpoint spacing of $0.05 \times 2\pi \unicode{x212B}^{-1}$.
*C19* is the default pseudopotential library for CASTEP, which has higher accuracy than the *QC5* used for searching, but it requires higher cut off energy.
The results are shown in the figure below.


```{code-cell} ipython3
:tags: [hide-input]

input_names = list(Path("lin2/refine-run1/good_castep/").glob("*.res"))
df_dft_refine = load_dataset(input_names)

input_names = list(Path("lin2/refine-run1-m3gnet/good_castep/").glob("*.res"))
df_m3g_refine = load_dataset(input_names)

ax = df_dft_refine.plot.scatter('volume_per_atom', 'energy_per_atom', label='CASTEP+QC5->C19')
ax = df_m3g_refine.plot.scatter('volume_per_atom', 'energy_per_atom',  xlim=(9, 20), label='M3GNet->C19', ax=ax, color='C1')
ax.set_ylim(df_dft_refine.energy_per_atom.min() - 0.01, df_dft_refine.energy_per_atom.min() + 0.3)
ax.legend(loc=1)
ax.set_ylabel('Energy per atom (eV)')
ax.set_xlabel(r'Volume per atom ($\mathrm{\AA^3}$)')
add_labels(ax)
```

It is clear that the structure from M3GNet are quite high in energy.
Relaxing these structures with DFT did change the geometry slightly, 
but the main structural features remain the same. 
Clearly, using M3GNet to search for this particularly composition, $\ce{LiN2}$, would not yield any fruitful results.

Are the new $LiN2$ structure found by DFT thermodynamically stable? 
It turns out the phase diagram of Li-N has indeed been studied previously:

- Shen, Y., Oganov, A., Qian, G. et al. Novel lithium-nitrogen compounds at ambient and high pressures. Sci Rep 5, 14204 (2015). https://doi.org/10.1038/srep14204

Where the same $\ce{LiN2}$ structure has been reported. 
These structures are not included in the Materials Project dataset, which probably explains why the model does not do well in the Li-N chemical space.


## Extras - Machine Learning Potential accelerated search for the Li-N composition space

Using M3GNet for geometry optimisation would certainly miss the $\ce{LiN2}$ ground state. This is not surprising because there is no similar structures in the training set. 

But can MLPs accelerate AIRSS? A search of Li-N phase space using the recent developed [Ephemeral data derived potentials](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.014102) can reproduce the convex hull of Shen et al 2015 above at ambient pressure, but there are also some intriguing new phases found, including $\ce{Li3N4}$  🤔🤔.

In this example, the potentials are iteratively built using DFT energies of randomly generated structures. It still at least for now we still need to pay the price of DFT. The training data involves ~ 60k DFT single point energy of small unit cells. This might sound a lot, they are roughly similar to the cost of just 1200 geometry optimisation. The potential built should be able to cover the entire composition space and allow sampling of larger unit cells.

Could one M3GNet make work for the Li-N if these training structures are included in the training? Or perhaps during fine-tuning?

![](hull-li-n.png)
