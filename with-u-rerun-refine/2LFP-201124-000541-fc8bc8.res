TITL 2LFP-201124-000541-fc8bc8 0.005 177.921 -95.6620 0.00 0.00 14 (I-4) n - 1
REM ALGO Normal
REM EDIFF 1.4e-05
REM EDIFFG -0.03
REM ENCUT 520
REM ENMAX 520.0
REM GGA Pe
REM IBRION -1
REM ICHARG 0
REM ISIF 3
REM ISMEAR 0
REM ISPIN 2
REM ISTART 1
REM KPOINT_BSE -1 0 0 0
REM LAECHG True
REM LASPH True
REM LCHARG True
REM LDAU True
REM LDAUJ [0.0, 0.0, 0.0, 0.0]
REM LDAUL [0, 2, 0, 0]
REM LDAUPRINT 1
REM LDAUTYPE 2
REM LDAUU [0.0, 4.0, 0.0, 0.0]
REM LMAXMIX 4
REM LORBIT 11
REM LREAL False
REM LVHAR True
REM LWAVE False
REM MAGMOM [0.001, 0.001, 3.699, 3.699, 0.006, 0.006, 0.031, 0.029, 0.03, 0.029, 0.029, 0.03, 0.029, 0.031]
REM NELM 200
REM NELMIN 6
REM NSW 0
REM POTIM 0.6
REM PREC Accurate
REM SIGMA 0.05
REM @module pymatgen.io.vasp.inputs
REM @class Incar
REM kpoint grid [[6, 6, 3]]
REM potcar symbols ['Li_sv', 'Fe_pv', 'P', 'O']
REM potcar functional PBE
CELL 1.0 4.809566     4.809566     7.691594     90.000000    90.000000    90.000000   
LATT -1
SFAC Li Fe P O
Li   1       0.0000000000000      0.0000000000000      0.0000000000000 1.0
Li   1       0.5000000000000      0.5000000000000      0.5000000000000 1.0
Fe   2       0.0000000000000      0.0000000000000      0.5000000000000 1.0
Fe   2       0.5000000000000      0.5000000000000      0.0000000000000 1.0
P    3       0.5000000000000      0.0000000000000      0.2499840000000 1.0
P    3       0.0000000000000      0.5000000000000      0.7500160000000 1.0
O    4       0.6872050000000      0.1893650000000      0.3651420000000 1.0
O    4       0.3106460000000      0.1871900000000      0.1348490000000 1.0
O    4       0.3127950000000      0.8106350000000      0.3651420000000 1.0
O    4       0.6893540000000      0.8128100000000      0.1348490000000 1.0
O    4       0.8106350000000      0.6872050000000      0.6348580000000 1.0
O    4       0.8128100000000      0.3106460000000      0.8651510000000 1.0
O    4       0.1893650000000      0.3127950000000      0.6348580000000 1.0
O    4       0.1871900000000      0.6893540000000      0.8651510000000 1.0
END
