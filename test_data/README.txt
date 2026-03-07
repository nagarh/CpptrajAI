Test files for cpptraj IDE
==========================
System : 10-residue polyalanine alpha-helix
Atoms  : 50 atoms (10 residues x 5 backbone atoms: N, CA, C, O, CB)
Frames : 200

Files
-----
test_topology.pdb   — single-frame PDB, use as topology (parm)
test_trajectory.pdb — multi-MODEL PDB, use as trajectory (trajin)

Example cpptraj script
----------------------
parm  test_topology.pdb
trajin test_trajectory.pdb

autoimage

rmsd backbone @CA,C,N,O first out rmsd.dat
atomicfluct rmsf @CA byres out rmsf.dat
radgyr rg @CA mass out rg.dat
hbond hbonds out hbond.dat avgout hbond_avg.dat
secstruct ss out secstruct.dat sumout secstruct_sum.dat

go
