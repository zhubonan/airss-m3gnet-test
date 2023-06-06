#!/usr/bin/env python
"""
Code for running relaxation using M3GNet

Relax a batch structures. Primiarily for running using `parallel`:

```
ls LiN2/run1/*-orig.cell | parallel -j 16 -n 30 python lfp-test.py LiN2/run1-m3gnet {} "2>&1" ">>" output-{%}
```

Will relax the structures matching pattern "LiN2/run1/*-orig.cell" and save the results in the folder `LiN2/run1-m3gnet`.
"""
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ["OMP_NUM_THREADS"]="1"    
os.environ["TF_NUM_INTRAOP_THREADS"]="1"    
os.environ["TF_NUM_INTEROP_THREADS"]="1"    


# In[2]:
import tensorflow as tf

#
import warnings
from pathlib import Path
from ase.io import read
from m3gnet.models import Relaxer
from disp.analysis.airssutils import RESFile
import ase
from pymatgen.io.ase import AseAtomsAdaptor

for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="tensorflow")

from tqdm import tqdm




def relax_one(cell, verbose=True, relaxer=None):
    """
    Relax one single structure
    
    Returns relaxed `ase.Atoms` object and the final energy per atom
    """
    
    if isinstance(cell, ase.Atoms):
        cell = AseAtomsAdaptor.get_structure(cell)
   
    if relaxer is None:
        relaxer = Relaxer()
    relax_results = relaxer.relax(cell, verbose=verbose)
    final_structure = relax_results['final_structure']
    final_energy = float(relax_results['trajectory'].energies[-1])
    return final_structure, final_energy


def process_one(cell, outdir='m3gnet-relaxed', relaxer=None):
    """Process a single atoms object and save the results as a SHELX file"""

    if relaxer is None:
        relaxer = Relaxer()

    structure, energy = relax_one(cell, verbose=False, relaxer=relaxer)
    fname = cell.info['fname'].replace('-orig', '')
    res = RESFile(structure, {'enthalpy': energy, 'label': fname, 'pressure': 0.})
    newname = Path(f'{outdir}/{fname}').with_suffix('.res')
    with open(newname, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(res.to_res_lines()))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) == 1:
        print("Usage: m3gnet-relax.py <outdir> <file1> <file2}.....")
    input_names = [Path(x) for x in sys.argv[2:]]
    outdir = sys.argv[1]

    cells = list(map(read, input_names))
    # Store the name of the files
    for atoms, name in zip(cells, input_names):
        atoms.info['fname'] = name.stem
    # Create the Relaxer object
    relaxer = Relaxer()
    [process_one(x, outdir, relaxer) for x in tqdm(cells)]



