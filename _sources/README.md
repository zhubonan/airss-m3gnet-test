# Test using M3GNet random structure searching


This repository contains results of testing [M3GNet](https://github.com/materialsvirtuallab/m3gnet) for random structure searching. 


A `m3gnet-relax.py` script is avaliable for relaxing structures:


```
python m3gnet-relax.py <outdir> <file1> <file2> ....
```

The signature is designed in away to allow easy parallelised executions using [parallel](https://www.gnu.org/software/parallel/). 

For example, to relax the DFT-relaxed LiFePO4 structure using M3GNet with 8 processes in parallel with 30 structures in each batch:

```
ls with-u-rerun/*.res | parallel -j 8 -n 30 python m3gnet-relax.py with-u-rerun-relaxed-m3gnet {}
```


The results can stored in the Jupyter notebooks. 

In addition, jupyter-book is used for building a website for hosting the results.
