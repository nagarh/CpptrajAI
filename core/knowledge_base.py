"""
cpptraj RAG knowledge base — built from the real CpptrajManual.pdf.

Pipeline:
  1. Extract text from PDF with pdfplumber (cached to cpptraj_manual_cache.json)
  2. Split into per-command chunks using section-header heuristics
  3. TF-IDF index (scikit-learn) for fast semantic-ish retrieval
  4. Thin structured command registry for the left-panel UI (unchanged look)
"""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

_HERE        = Path(__file__).parent.parent          # CPPTRAJ_Agent/
PDF_PATH     = _HERE / "CpptrajManual.pdf"
CACHE_PATH   = _HERE / "cpptraj_manual_cache.json"

# ─────────────────────────────────────────────────────────────────────────────
# CPPTRAJ COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

CPPTRAJ_COMMANDS = {
    # ── Setup / Input ────────────────────────────────────────────────────────
    "parm":               {"category": "Setup",        "title": "Load Topology (parm)",                    "description": "Load a topology/parameter file (.prmtop, .psf, .gro, .pdb). Must be the first command.", "syntax": "parm <filename> [<tag>] [nobondsearch]"},
    "trajin":             {"category": "Setup",        "title": "Load Trajectory (trajin)",                "description": "Load trajectory file(s). Multiple trajin statements concatenate frames. Use start/stop/offset to sub-sample.", "syntax": "trajin <filename> [start] [stop|last] [offset]"},
    "reference":          {"category": "Setup",        "title": "Load Reference (reference)",              "description": "Load a reference structure used by rmsd, align, nativecontacts.", "syntax": "reference <filename> [<frame>] [<tag>]"},
    "activeref":          {"category": "Setup",        "title": "Set Active Reference (activeref)",        "description": "Set the active reference structure by tag.", "syntax": "activeref <tag>"},
    "createcrd":          {"category": "Setup",        "title": "Create COORDS Set (createcrd)",           "description": "Create an empty COORDS data set for in-memory trajectory storage.", "syntax": "createcrd <name>"},
    "createreservoir":    {"category": "Setup",        "title": "Create Reservoir (createreservoir)",      "description": "Create structure reservoir for REST simulation.", "syntax": "createreservoir <name> <filename> [<fmt>] [<mask>] [ene <set>] [temp <T>]"},
    "createset":          {"category": "Setup",        "title": "Create Data Set (createset)",             "description": "Create a new data set with specified values.", "syntax": "createset name <name> type <type> [values <v1>,<v2>,...] [<range>]"},
    "loadcrd":            {"category": "Setup",        "title": "Load COORDS (loadcrd)",                   "description": "Load trajectory into a named COORDS data set for later use.", "syntax": "loadcrd <filename> [<fmt>] [<mask>] name <setname>"},
    "loadtraj":           {"category": "Setup",        "title": "Load Trajectory (loadtraj)",              "description": "Load trajectory (alias for trajin inside scripts).", "syntax": "loadtraj <filename> [<fmt>] [<mask>]"},
    "readdata":           {"category": "Setup",        "title": "Read Data (readdata)",                    "description": "Read data from file into data sets for analysis.", "syntax": "readdata <filename> [as <fmt>] [name <name>] [index <col>]"},
    "go":                 {"category": "Setup",        "title": "Execute (go)",                            "description": "Execute all queued commands. Required at end of every script.", "syntax": "go"},
    # ── Output ───────────────────────────────────────────────────────────────
    "trajout":            {"category": "Output",       "title": "Write Trajectory (trajout)",              "description": "Write processed trajectory to a new file. Format auto-detected from extension.", "syntax": "trajout <filename> [format] [nobox]"},
    "outtraj":            {"category": "Output",       "title": "Write Frames On-the-fly (outtraj)",       "description": "Write frames to trajectory file during processing.", "syntax": "outtraj <filename> [<fmt>] [<mask>] [nobox] [onlyframes <range>]"},
    "crdout":             {"category": "Output",       "title": "Write COORDS Set (crdout)",               "description": "Write a COORDS data set to a trajectory file.", "syntax": "crdout <crdset> <filename> [<fmt>] [<mask>]"},
    "parmwrite":          {"category": "Output",       "title": "Write Topology (parmwrite)",              "description": "Write topology to file in specified format.", "syntax": "parmwrite out <filename> [<fmt>] [<topology tag>]"},
    "datafile":           {"category": "Output",       "title": "Data File Options (datafile)",            "description": "Set output options for a data file.", "syntax": "datafile <filename> [<options>]"},
    "datafilter":         {"category": "Output",       "title": "Filter Data (datafilter)",                "description": "Filter data sets by criteria and write to file.", "syntax": "datafilter <dataset> min <min> max <max> [out <file>]"},
    "dataset":            {"category": "Output",       "title": "Data Set Operations (dataset)",           "description": "Perform operations on data sets: legend, makexy, etc.", "syntax": "dataset {legend <legend> <set> | makexy <X> <Y> name <out> | ...}"},
    "flatten":            {"category": "Output",       "title": "Flatten Data (flatten)",                  "description": "Flatten multi-dimensional data sets to 1D.", "syntax": "flatten <dataset> [out <file>]"},
    "precision":          {"category": "Output",       "title": "Output Precision (precision)",            "description": "Set output precision for data files.", "syntax": "precision <file> <width> [<digits>]"},
    "selectds":           {"category": "Output",       "title": "Select Data Sets (selectds)",             "description": "Select data sets matching a string pattern.", "syntax": "selectds <selection>"},
    # ── Manipulation / Actions ───────────────────────────────────────────────
    "autoimage":          {"category": "Manipulation", "title": "Fix PBC Imaging (autoimage)",             "description": "Re-image molecules across periodic boundaries back into the primary unit cell. Always strip :WAT first.", "syntax": "autoimage [familiar] [byres|bymol] [anchor <mask>]"},
    "center":             {"category": "Manipulation", "title": "Center System (center)",                  "description": "Translate coordinates so that specified atoms are at the origin or box center.", "syntax": "center [<mask>] [origin] [mass]"},
    "strip":              {"category": "Manipulation", "title": "Strip Atoms (strip)",                     "description": "Remove atoms/residues/molecules from the trajectory.", "syntax": "strip <mask>"},
    "align":              {"category": "Manipulation", "title": "Align Trajectory (align)",                "description": "Rotate and translate frames to least-squares-fit selected atoms to a reference. Modifies coordinates.", "syntax": "align [<mask>] [ref <tag>|reference|first] [mass]"},
    "image":              {"category": "Manipulation", "title": "Image Molecules (image)",                 "description": "Image molecules into primary unit cell. Use autoimage for automatic imaging.", "syntax": "image [familiar] [bymol|byres|byatom] [<mask>] [origin] [center]"},
    "unwrap":             {"category": "Manipulation", "title": "Unwrap Trajectory (unwrap)",              "description": "Unwrap trajectory to remove periodic boundary jumps.", "syntax": "unwrap [<mask>] [center] [bymol|byres]"},
    "unstrip":            {"category": "Manipulation", "title": "Restore Stripped Atoms (unstrip)",        "description": "Restore previously stripped atoms back to the system.", "syntax": "unstrip"},
    "translate":          {"category": "Manipulation", "title": "Translate Coordinates (translate)",       "description": "Translate coordinates by a vector.", "syntax": "translate [<mask>] [x <dx>] [y <dy>] [z <dz>]"},
    "rotate":             {"category": "Manipulation", "title": "Rotate Coordinates (rotate)",             "description": "Rotate coordinates around an axis.", "syntax": "rotate [<mask>] {axis <x,y,z> degrees <d> | x|y|z <deg>}"},
    "scale":              {"category": "Manipulation", "title": "Scale Coordinates (scale)",               "description": "Scale coordinates by a factor along x/y/z.", "syntax": "scale [<mask>] [x <fx>] [y <fy>] [z <fz>]"},
    "box":                {"category": "Manipulation", "title": "Set Box Dimensions (box)",                "description": "Set or modify unit cell box dimensions.", "syntax": "box [x <x>] [y <y>] [z <z>] [alpha <a>] [beta <b>] [gamma <g>] [nobox]"},
    "closest":            {"category": "Manipulation", "title": "Keep Closest Solvent (closest)",          "description": "Keep N closest solvent molecules to solute, remove the rest.", "syntax": "closest <N> <solvent_mask> [noimage] [first|oxygen] [name <name>]"},
    "addatom":            {"category": "Manipulation", "title": "Add Atom (addatom)",                      "description": "Add atoms to the topology.", "syntax": "addatom {bond <mask> | nobond} <name> <type> <charge> <mass> [<coords>]"},
    "atommap":            {"category": "Manipulation", "title": "Map Atoms (atommap)",                     "description": "Map atoms between two structures/topologies.", "syntax": "atommap <ref> <target> [mapout <file>] [maponly]"},
    "catcrd":             {"category": "Manipulation", "title": "Concatenate COORDS (catcrd)",             "description": "Concatenate multiple COORDS data sets.", "syntax": "catcrd [crdset <set1>] [crdset <set2>] ... name <output>"},
    "change":             {"category": "Manipulation", "title": "Change Topology Properties (change)",     "description": "Change topology atom/residue names, types, or other properties.", "syntax": "change {resname from <old> to <new> | atomname from <old> to <new> | ...}"},
    "charge":             {"category": "Manipulation", "title": "Print Total Charge (charge)",             "description": "Print total charge for atom selection.", "syntax": "charge [<mask>]"},
    "checkchirality":     {"category": "Manipulation", "title": "Check Chirality (checkchirality)",        "description": "Check chirality of chiral centers.", "syntax": "checkchirality [<mask>] [out <file>]"},
    "combinecrd":         {"category": "Manipulation", "title": "Combine COORDS (combinecrd)",             "description": "Combine two COORDS sets into one.", "syntax": "combinecrd <crdset1> <crdset2> name <output>"},
    "comparetop":         {"category": "Manipulation", "title": "Compare Topologies (comparetop)",         "description": "Compare two topology files.", "syntax": "comparetop [parm1 <tag>] [parm2 <tag>]"},
    "crdaction":          {"category": "Manipulation", "title": "Apply Action to COORDS (crdaction)",      "description": "Apply an action to a COORDS data set.", "syntax": "crdaction <crdset> <action> [<action_args>]"},
    "crdfluct":           {"category": "Manipulation", "title": "COORDS Fluctuations (crdfluct)",          "description": "Calculate fluctuations of a COORDS data set.", "syntax": "crdfluct <crdset> [<mask>] [out <file>] [byres] [bfactor]"},
    "crdtransform":       {"category": "Manipulation", "title": "Transform COORDS (crdtransform)",         "description": "Apply coordinate transformation to a COORDS set.", "syntax": "crdtransform <crdset> [<xform_args>]"},
    "dihedralscan":       {"category": "Manipulation", "title": "Dihedral Scan (dihedralscan)",            "description": "Scan dihedral angles to generate conformations.", "syntax": "dihedralscan [<mask>] [rseed <seed>] [out <file>] [outtraj <file>]"},
    "emin":               {"category": "Manipulation", "title": "Energy Minimization (emin)",              "description": "Energy minimization using internal force field.", "syntax": "emin [<mask>] [nstep <N>] [out <file>] [step <step>]"},
    "fiximagedbonds":     {"category": "Manipulation", "title": "Fix Imaged Bonds (fiximagedbonds)",       "description": "Fix broken bonds across periodic boundaries.", "syntax": "fiximagedbonds [<mask>]"},
    "fixatomorder":       {"category": "Manipulation", "title": "Fix Atom Order (fixatomorder)",           "description": "Reorder atoms to match topology.", "syntax": "fixatomorder [<mask>] [outprefix <prefix>]"},
    "graft":              {"category": "Manipulation", "title": "Graft Coordinates (graft)",               "description": "Graft coordinates from one structure onto another.", "syntax": "graft [src <mask>] [tgt <mask>] [srcframe <N>] [mass]"},
    "hmassrepartition":   {"category": "Manipulation", "title": "H-mass Repartition (hmassrepartition)",   "description": "Hydrogen mass repartitioning for longer MD timesteps.", "syntax": "hmassrepartition [<mask>] [factor <f>]"},
    "lessplit":           {"category": "Manipulation", "title": "Split LES Trajectory (lessplit)",         "description": "Split LES trajectory into individual replicas.", "syntax": "lessplit [out <prefix>] [<fmt>]"},
    "makestructure":      {"category": "Manipulation", "title": "Build Structure (makestructure)",         "description": "Build structure using idealized geometry.", "syntax": "makestructure <sstype>:<res_range>[,...] [out <prefix>]"},
    "minimage":           {"category": "Manipulation", "title": "Minimum Image (minimage)",                "description": "Apply minimum image convention for periodic distance.", "syntax": "minimage [<name>] <mask1> <mask2> [out <file>]"},
    "molinfo":            {"category": "Manipulation", "title": "Molecule Info (molinfo)",                 "description": "Print molecular information for atom mask.", "syntax": "molinfo [<mask>] [<topology tag>]"},
    "parmbox":            {"category": "Manipulation", "title": "Set Topology Box (parmbox)",              "description": "Set periodic box dimensions in topology.", "syntax": "parmbox {x <x> y <y> z <z> [alpha <a> beta <b> gamma <g>] | nobox}"},
    "parminfo":           {"category": "Manipulation", "title": "Topology Info (parminfo)",                "description": "Print topology information summary.", "syntax": "parminfo [<mask>] [<topology tag>]"},
    "parmstrip":          {"category": "Manipulation", "title": "Strip Topology (parmstrip)",              "description": "Strip atoms from topology file permanently.", "syntax": "parmstrip <mask> [<topology tag>]"},
    "permutedihedrals":   {"category": "Manipulation", "title": "Permute Dihedrals (permutedihedrals)",    "description": "Randomly permute dihedral angles.", "syntax": "permutedihedrals [<mask>] [rseed <seed>] [out <file>]"},
    "prepareforleap":     {"category": "Manipulation", "title": "Prepare for LEaP (prepareforleap)",       "description": "Prepare structure for LEaP (add missing atoms, fix naming).", "syntax": "prepareforleap [<mask>] [out <file>] [pdbout <file>]"},
    "randomizeions":      {"category": "Manipulation", "title": "Randomize Ions (randomizeions)",          "description": "Randomly swap ions with solvent molecules.", "syntax": "randomizeions <ion_mask> [by <solvent_mask>] [around <solute_mask>] [min <d>] [rseed <s>]"},
    "remap":              {"category": "Manipulation", "title": "Remap Atom Order (remap)",                "description": "Remap atom ordering to match a reference.", "syntax": "remap [<mask>] <reference>"},
    "replicatecell":      {"category": "Manipulation", "title": "Replicate Unit Cell (replicatecell)",     "description": "Replicate the unit cell in 3D.", "syntax": "replicatecell [<mask>] [out <prefix>] {all | dir X Y Z}"},
    "resinfo":            {"category": "Manipulation", "title": "Residue Info (resinfo)",                  "description": "Print residue information: resid, resname, atom count, etc.", "syntax": "resinfo [<mask>] [<topology tag>]"},
    "rotatedihedral":     {"category": "Manipulation", "title": "Rotate Dihedral (rotatedihedral)",        "description": "Rotate a specific dihedral angle to a target value.", "syntax": "rotatedihedral [<mask>] res <r> type <phi|psi|chi1...> val <degrees>"},
    "scale":              {"category": "Manipulation", "title": "Scale Coordinates (scale)",               "description": "Scale coordinates by a factor along x/y/z.", "syntax": "scale [<mask>] [x <fx>] [y <fy>] [z <fz>]"},
    "select":             {"category": "Manipulation", "title": "Select Atoms (select)",                   "description": "Select atoms by mask and print information.", "syntax": "select <mask>"},
    "sequence":           {"category": "Manipulation", "title": "Print Sequence (sequence)",               "description": "Print amino acid or nucleic acid sequence.", "syntax": "sequence [<mask>] [<topology tag>]"},
    "setvelocity":        {"category": "Manipulation", "title": "Set Velocities (setvelocity)",            "description": "Assign velocities from Maxwell-Boltzmann distribution.", "syntax": "setvelocity [<mask>] [temp <T>] [rseed <seed>]"},
    "solvent":            {"category": "Manipulation", "title": "Define Solvent (solvent)",                "description": "Define solvent molecules in topology.", "syntax": "solvent [<mask>] [<topology tag>]"},
    "splitcoords":        {"category": "Manipulation", "title": "Split COORDS (splitcoords)",              "description": "Split COORDS set into separate sets by frame.", "syntax": "splitcoords <crdset> [<range>] name <prefix>"},
    "updateparameters":   {"category": "Manipulation", "title": "Update Parameters (updateparameters)",    "description": "Update force field parameters in topology.", "syntax": "updateparameters {<bond_args>|<angle_args>|<dih_args>}"},
    "bondparminfo":       {"category": "Manipulation", "title": "Bond Parameter Info (bondparminfo)",      "description": "Print bond parameter information.", "syntax": "bondparminfo [<mask>] [<topology tag>]"},
    # ── Analysis ─────────────────────────────────────────────────────────────
    "rmsd":               {"category": "Analysis",     "title": "RMSD (rmsd)",                            "description": "Calculate frame-by-frame RMSD of atoms relative to a reference. Use @CA,C,N,O for backbone. Most common MD analysis.", "syntax": "rmsd [<name>] [<mask>] [ref <tag>|first|reference] [out <file>] [nofit] [mass] [perres]"},
    "atomicfluct":        {"category": "Analysis",     "title": "RMSF (atomicfluct)",                     "description": "Per-atom or per-residue root mean square fluctuation (B-factors). Use byres for per-residue.", "syntax": "atomicfluct [<name>] [<mask>] [out <file>] [byres] [byatom] [bfactor]"},
    "radgyr":             {"category": "Analysis",     "title": "Radius of Gyration (radgyr)",            "description": "Calculate radius of gyration — measures compactness. Always use mass keyword.", "syntax": "radgyr [<name>] [<mask>] [out <file>] [mass] [tensor]"},
    "hbond":              {"category": "Analysis",     "title": "Hydrogen Bonds (hbond)",                 "description": "Detect and track hydrogen bonds. Default: dist ≤ 3.5 Å, angle ≥ 135°. Use avgout for statistics.", "syntax": "hbond [<name>] [<mask>] [out <file>] [avgout <file>] [dist <A>] [angle <deg>] [series]"},
    "secstruct":          {"category": "Analysis",     "title": "Secondary Structure (secstruct)",        "description": "Assign secondary structure using DSSP algorithm. H=helix, E=strand, T=turn, C=coil.", "syntax": "secstruct [<name>] [<mask>] [out <file>] [sumout <file>]"},
    "dssp":               {"category": "Analysis",     "title": "DSSP Secondary Structure (dssp)",        "description": "DSSP secondary structure assignment — alias for secstruct.", "syntax": "dssp [<name>] [<mask>] [out <file>] [sumout <file>]"},
    "cluster":            {"category": "Analysis",     "title": "Clustering (cluster)",                   "description": "Cluster trajectory frames by structural similarity. Use sieve for large trajectories.", "syntax": "cluster [<name>] [<mask>] [hieragglo|kmeans|dbscan] [epsilon <val>] [clusters <N>] [out <file>] [summary <file>] [repout <prefix>] [repfmt pdb]"},
    "distance":           {"category": "Analysis",     "title": "Distance (distance)",                    "description": "Calculate distance between two atom masks (center-of-mass by default).", "syntax": "distance [<name>] <mask1> <mask2> [out <file>] [noimage] [geom]"},
    "angle":              {"category": "Analysis",     "title": "Angle (angle)",                          "description": "Calculate angle between three atoms or groups. mask2 is the vertex.", "syntax": "angle [<name>] <mask1> <mask2> <mask3> [out <file>]"},
    "dihedral":           {"category": "Analysis",     "title": "Dihedral (dihedral)",                    "description": "Calculate dihedral (torsion) angle from four atoms. Output in −180 to +180 degrees.", "syntax": "dihedral [<name>] <mask1> <mask2> <mask3> <mask4> [out <file>]"},
    "multidihedral":      {"category": "Analysis",     "title": "Backbone Dihedrals (multidihedral)",     "description": "Calculate phi, psi, omega, chi1-chi4 for all or selected residues.", "syntax": "multidihedral [phi] [psi] [omega] [chin] [<mask>] [out <file>]"},
    "phipsi":             {"category": "Analysis",     "title": "Phi/Psi Ramachandran (phipsi)",          "description": "Calculate Ramachandran phi/psi angles for residues.", "syntax": "phipsi [<mask>] [out <file>] [name <name>] [resrange <range>]"},
    "surf":               {"category": "Analysis",     "title": "SASA (surf)",                            "description": "Calculate solvent-accessible surface area using LCPO algorithm. 1.4 Å probe.", "syntax": "surf [<name>] [<mask>] [out <file>] [solvradius <val>]"},
    "molsurf":            {"category": "Analysis",     "title": "MSMS SASA (molsurf)",                    "description": "MSMS/molsurf solvent accessible surface area.", "syntax": "molsurf [<name>] [<mask>] [out <file>] [probe <r>]"},
    "nativecontacts":     {"category": "Analysis",     "title": "Native Contacts (nativecontacts)",       "description": "Calculate fraction of native contacts (Q-value) relative to a reference structure.", "syntax": "nativecontacts [<name>] [<mask>] [ref <tag>|reference] [out <file>] [distance <cutoff>]"},
    "contacts":           {"category": "Analysis",     "title": "Contacts (contacts)",                    "description": "Calculate number of contacts. Legacy command — prefer nativecontacts.", "syntax": "contacts [first|reference|ref <ref>] [byresidue] [out <file>] [<mask>]"},
    "density":            {"category": "Analysis",     "title": "Density Profile (density)",              "description": "Calculate number or mass density along an axis. Useful for membrane systems.", "syntax": "density [<name>] [<mask>] [out <file>] [x|y|z] [delta <dx>] [number|mass|electron]"},
    "diffusion":          {"category": "Analysis",     "title": "Diffusion / MSD (diffusion)",            "description": "Calculate mean square displacement and diffusion coefficient. D = slope of MSD / 6.", "syntax": "diffusion [<name>] [<mask>] [out <file>] [time <dt>] [diffout <file>]"},
    "stfcdiffusion":      {"category": "Analysis",     "title": "STFC Diffusion (stfcdiffusion)",         "description": "Diffusion using STFC method for charged particles.", "syntax": "stfcdiffusion [<mask>] [out <file>] [time <dt>] [x|y|z|xy|xz|yz|xyz]"},
    "calcdiffusion":      {"category": "Analysis",     "title": "Calc Diffusion Coefficient (calcdiffusion)", "description": "Calculate diffusion coefficient from MSD data set.", "syntax": "calcdiffusion <msd_set> [out <file>] [time <ts>]"},
    "watershell":         {"category": "Analysis",     "title": "Water Shell (watershell)",               "description": "Count water molecules in first and second solvation shells around a solute.", "syntax": "watershell [<name>] <mask> [out <file>] [lower <A>] [upper <A>]"},
    "radial":             {"category": "Analysis",     "title": "Radial Distribution Function (radial)",  "description": "Calculate radial distribution function (RDF) g(r).", "syntax": "radial [out <file>] <spacing> <maximum> <solvent_mask> [<solute_mask>] [noimage]"},
    "volmap":             {"category": "Analysis",     "title": "Volumetric Map (volmap)",                "description": "Generate 3D volumetric density map (.dx file, viewable in VMD).", "syntax": "volmap <filename> [<mask>] [size <dx> <dy> <dz>] [center <mask>]"},
    "grid":               {"category": "Analysis",     "title": "3D Density Grid (grid)",                 "description": "Calculate 3D density grid.", "syntax": "grid <filename> <dx> <dy> <dz> [origin] [<mask>] [box]"},
    "pucker":             {"category": "Analysis",     "title": "Ring Pucker (pucker)",                   "description": "Calculate Cremer-Pople ring pucker parameters for sugars/nucleic acids.", "syntax": "pucker [<name>] <m1> <m2> <m3> <m4> <m5> [<m6>] [out <file>] [amplitude] [theta]"},
    "multipucker":        {"category": "Analysis",     "title": "Multi Ring Pucker (multipucker)",        "description": "Calculate ring pucker for multiple residues.", "syntax": "multipucker [<mask>] [out <file>] [amplitude] [theta]"},
    "matrix":             {"category": "Analysis",     "title": "Covariance Matrix (matrix)",             "description": "Build covariance or correlation matrix — first step for PCA.", "syntax": "matrix covar [<name>] [<mask>] [out <file>]"},
    "diagmatrix":         {"category": "Analysis",     "title": "Diagonalize Matrix (diagmatrix)",        "description": "Diagonalize a matrix to get eigenvalues and eigenvectors.", "syntax": "diagmatrix <matrixset> [out <evecfile>] [vecs <N>] [reduce] [mass <mask>]"},
    "projection":         {"category": "Analysis",     "title": "PCA Projection (projection)",            "description": "Project trajectory onto eigenvectors from matrix/analyze modes for PCA.", "syntax": "projection [<name>] evecvecs <data> [<mask>] [out <file>] [beg <n>] [end <n>]"},
    "modes":              {"category": "Analysis",     "title": "Normal Modes (modes)",                   "description": "Analyze normal modes from diagonalized matrix: fluct, displ, corr, eigenval, trajout.", "syntax": "modes {fluct|displ|corr|eigenval|trajout} name <modesname> [beg <b>] [end <e>] [out <file>]"},
    "tica":               {"category": "Analysis",     "title": "TICA (tica)",                            "description": "Time-lagged independent component analysis.", "syntax": "tica {crdset <COORDS>|data <sets>} [lag <lag>] [nvecs <N>] [out <file>]"},
    "atomiccorr":         {"category": "Analysis",     "title": "Atomic Correlation (atomiccorr)",        "description": "Atomic correlation matrix between atom displacements.", "syntax": "atomiccorr [out <file>] [cut <cut>] [<mask>] [datasave <set>]"},
    "rms2d":              {"category": "Analysis",     "title": "Pairwise RMSD Matrix (rms2d)",           "description": "Pairwise RMSD matrix between all frame pairs.", "syntax": "rms2d [<name>] [<mask>] [out <file>] [mass] [nofit] [reftraj <traj>]"},
    "rmsavgcorr":         {"category": "Analysis",     "title": "RMSD Running Average Correlation (rmsavgcorr)", "description": "Correlation of running-average RMSD vs window size.", "syntax": "rmsavgcorr [<mask>] [out <file>] [mass]"},
    "symmrmsd":           {"category": "Analysis",     "title": "Symmetric RMSD (symmrmsd)",              "description": "RMSD with symmetry correction for equivalent atoms.", "syntax": "symmrmsd [<name>] [<mask>] [ref <ref>|first] [out <file>] [remap]"},
    "dihedralrms":        {"category": "Analysis",     "title": "Dihedral RMSD (dihedralrms)",            "description": "RMSD of dihedral angles between frames.", "syntax": "dihedralrms [<mask>] [out <file>] [mass] [nofit]"},
    "clusterdihedral":    {"category": "Analysis",     "title": "Dihedral Clustering (clusterdihedral)",  "description": "Cluster by dihedral angles.", "syntax": "clusterdihedral [<mask>] [out <file>] [clusterout <prefix>] [...dihedrals...]"},
    "average":            {"category": "Analysis",     "title": "Average Structure (average)",            "description": "Compute average structure over trajectory frames.", "syntax": "average [<name>] <filename> [<fmt>] [<mask>] [start <s>] [stop <e>] [offset <o>]"},
    "avgcoord":           {"category": "Analysis",     "title": "Average Coordinates (avgcoord)",         "description": "Average coordinates for each atom over trajectory.", "syntax": "avgcoord [<name>] [<mask>] [out <file>]"},
    "avgbox":             {"category": "Analysis",     "title": "Average Box (avgbox)",                   "description": "Compute average box dimensions over trajectory.", "syntax": "avgbox [<name>] [out <file>]"},
    "bounds":             {"category": "Analysis",     "title": "Bounding Box (bounds)",                  "description": "Calculate bounding box around atoms.", "syntax": "bounds [<name>] [<mask>] [out <file>] [dx <dx>] [offset <offset>]"},
    "principal":          {"category": "Analysis",     "title": "Principal Axes (principal)",             "description": "Calculate principal axes and moments of inertia.", "syntax": "principal [<name>] [<mask>] [out <file>] [dorotation] [mass]"},
    "dipole":             {"category": "Analysis",     "title": "Dipole Moment (dipole)",                 "description": "Calculate dipole moment of selection.", "syntax": "dipole [<name>] [<mask>] [out <file>] [<grid_options>]"},
    "volume":             {"category": "Analysis",     "title": "Unit Cell Volume (volume)",              "description": "Calculate unit cell volume over trajectory.", "syntax": "volume [<name>] [out <file>]"},
    "temperature":        {"category": "Analysis",     "title": "Temperature (temperature)",              "description": "Calculate instantaneous temperature from velocities.", "syntax": "temperature [<name>] [<mask>] [out <file>] [frame]"},
    "energy":             {"category": "Analysis",     "title": "Energy (energy)",                        "description": "Calculate energy using internal force field (bond, angle, dihedral, VdW, electrostatic).", "syntax": "energy [<mask>] [out <file>] [bond] [angle] [dih] [vdw] [elec]"},
    "esander":            {"category": "Analysis",     "title": "Energy via Sander (esander)",            "description": "Calculate energy using sander AMBER engine.", "syntax": "esander [<mask>] [out <file>] [igb <igb>] [cut <cut>]"},
    "enedecomp":          {"category": "Analysis",     "title": "Energy Decomposition (enedecomp)",       "description": "Energy decomposition per residue.", "syntax": "enedecomp [<mask>] [out <file>] [cut <cut>]"},
    "pairwise":           {"category": "Analysis",     "title": "Pairwise Energy (pairwise)",             "description": "Pairwise energy decomposition between residues.", "syntax": "pairwise [<mask>] [out <file>] [cut <cut>] [cuteelec <c>] [cutevdw <c>]"},
    "lie":                {"category": "Analysis",     "title": "Linear Interaction Energy (lie)",        "description": "Linear interaction energy calculation.", "syntax": "lie <mask1> [<mask2>] [out <file>] [elec <scale>] [vdw <scale>]"},
    "ti":                 {"category": "Analysis",     "title": "Thermodynamic Integration (ti)",         "description": "Thermodynamic integration (TI) free energy calculation.", "syntax": "ti <dset0> [<dset1>...] {nq <n>|xvals <x>} [out <file>] [name <name>]"},
    "spam":               {"category": "Analysis",     "title": "SPAM (spam)",                            "description": "Solvation parameters from analysis of MD.", "syntax": "spam <site_file> [out <file>] [name <name>] [DG <dg>]"},
    "nastruct":           {"category": "Analysis",     "title": "Nucleic Acid Structure (nastruct)",      "description": "Nucleic acid structure parameters: base pairs, helical parameters.", "syntax": "nastruct [<name>] [resrange <range>] [naout <suffix>] [sscalc] [noheader]"},
    "jcoupling":          {"category": "Analysis",     "title": "J-coupling (jcoupling)",                 "description": "Calculate J-coupling constants from dihedral angles using Karplus equation.", "syntax": "jcoupling [<mask>] [kfile <karplus_file>] [out <file>]"},
    "ired":               {"category": "Analysis",     "title": "iRED NMR (ired)",                        "description": "iRED analysis of NMR order parameters.", "syntax": "ired [relax freq <MHz>] [order <o>] [orderparamfile <f>] [tstep <t>] [tcorr <t>] [out <f>]"},
    "rotdif":             {"category": "Analysis",     "title": "Rotational Diffusion (rotdif)",          "description": "Rotational diffusion analysis from NMR relaxation.", "syntax": "rotdif [out <file>] [rvecin <file>] [rseed <seed>] [nvecs <N>]"},
    "timecorr":           {"category": "Analysis",     "title": "Time Correlation (timecorr)",            "description": "Time correlation function of vectors.", "syntax": "timecorr vec1 <set> [vec2 <set>] [out <file>] [tstep <t>] [tcorr <t>]"},
    "vector":             {"category": "Analysis",     "title": "Vector (vector)",                        "description": "Calculate a vector between two masks over time.", "syntax": "vector [<name>] <mask1> <mask2> [out <file>] [ired]"},
    "multivector":        {"category": "Analysis",     "title": "Multi-vector (multivector)",             "description": "Calculate vectors for multiple residue pairs.", "syntax": "multivector [<mask>] [out <file>] [ired]"},
    "vectormath":         {"category": "Analysis",     "title": "Vector Math (vectormath)",               "description": "Math operations on vector data sets: dot product, cross product, etc.", "syntax": "vectormath vec1 <set> [vec2 <set>] {dotproduct|crossproduct|...} [out <file>]"},
    "velocityautocorr":   {"category": "Analysis",     "title": "Velocity Autocorrelation (velocityautocorr)", "description": "Velocity autocorrelation function (VACF).", "syntax": "velocityautocorr [<mask>] [out <file>] [tstep <t>] [maxlag <m>] [norm]"},
    "lipidorder":         {"category": "Analysis",     "title": "Lipid Order Parameters (lipidorder)",    "description": "Calculate lipid tail order parameters (Scd) for membrane systems.", "syntax": "lipidorder [<mask>] [out <file>] [scd] [unsat]"},
    "lipidscd":           {"category": "Analysis",     "title": "Lipid Scd (lipidscd)",                   "description": "Lipid Scd order parameter calculation.", "syntax": "lipidscd [<mask>] [out <file>]"},
    "areapermol":         {"category": "Analysis",     "title": "Area per Molecule (areapermol)",         "description": "Calculate area per molecule for lipid bilayers.", "syntax": "areapermol [<name>] [out <file>] [<mask>] [frame]"},
    "mindist":            {"category": "Analysis",     "title": "Min/Max Distance (mindist)",             "description": "Minimum and maximum distance between two masks.", "syntax": "mindist [<name>] <mask1> <mask2> [out <file>] [noimage]"},
    "pairdist":           {"category": "Analysis",     "title": "Pairwise Distance (pairdist)",           "description": "Pairwise distance histogram between all atom pairs.", "syntax": "pairdist [<name>] [<mask>] [out <file>] [delta <dx>] [max <max>]"},
    "hausdorff":          {"category": "Analysis",     "title": "Hausdorff Distance (hausdorff)",         "description": "Calculate Hausdorff distance between two masks.", "syntax": "hausdorff [<name>] <mask1> <mask2> [out <file>]"},
    "tordiff":            {"category": "Analysis",     "title": "Torsion Difference (tordiff)",           "description": "Torsion angle difference between two structures.", "syntax": "tordiff [<mask>] [out <file>] [ref <ref>]"},
    "autocorr":           {"category": "Analysis",     "title": "Autocorrelation (autocorr)",             "description": "Autocorrelation function of a data set.", "syntax": "autocorr <dataset> [out <file>] [lagmax <max>] [norm] [direct]"},
    "crosscorr":          {"category": "Analysis",     "title": "Cross-correlation (crosscorr)",          "description": "Cross-correlation between two data sets.", "syntax": "crosscorr <set1> <set2> [out <file>] [lagmax <max>] [norm] [direct]"},
    "lifetime":           {"category": "Analysis",     "title": "Lifetime Analysis (lifetime)",           "description": "Lifetime analysis of hydrogen bonds or contacts.", "syntax": "lifetime <dataset> [out <file>] [window <w>] [cut <cut>] [name <name>]"},
    "runningavg":         {"category": "Analysis",     "title": "Running Average (runningavg)",           "description": "Running average (sliding window) of a data set.", "syntax": "runningavg <dataset> [out <file>] [window <w>]"},
    "integrate":          {"category": "Analysis",     "title": "Integrate (integrate)",                  "description": "Integrate a data set using the trapezoidal rule.", "syntax": "integrate <dataset> [out <file>]"},
    "slope":              {"category": "Analysis",     "title": "Slope / Linear Fit (slope)",             "description": "Calculate slope of a data set by linear fit.", "syntax": "slope <dataset> [out <file>]"},
    "regress":            {"category": "Analysis",     "title": "Linear Regression (regress)",            "description": "Linear regression of a data set.", "syntax": "regress <dataset> [out <file>] [results <file>]"},
    "curvefit":           {"category": "Analysis",     "title": "Curve Fitting (curvefit)",               "description": "Fit data to a functional form.", "syntax": "curvefit <function> <dataset> [out <file>] [results <file>] [nofit]"},
    "kde":                {"category": "Analysis",     "title": "KDE (kde)",                              "description": "Kernel density estimation of a data set.", "syntax": "kde <dataset> [out <file>] [bandwidth <bw>] [bins <N>]"},
    "fft":                {"category": "Analysis",     "title": "FFT (fft)",                              "description": "Fast Fourier Transform of a data set.", "syntax": "fft <dataset> [out <file>] [dt <timestep>] [fftout <file>]"},
    "wavelet":            {"category": "Analysis",     "title": "Wavelet Analysis (wavelet)",             "description": "Wavelet analysis of trajectory data.", "syntax": "wavelet [<mask>] [out <file>] [type <wavelet>] [nb <N>]"},
    "filter":             {"category": "Analysis",     "title": "Filter Frames (filter)",                 "description": "Filter frames based on dataset value criteria.", "syntax": "filter <dataset> min <min> max <max>"},
    "divergence":         {"category": "Analysis",     "title": "KL Divergence (divergence)",             "description": "Calculate KL divergence between two distributions.", "syntax": "divergence <set1> <set2> [out <file>]"},
    "lowestcurve":        {"category": "Analysis",     "title": "Lowest Free Energy Curve (lowestcurve)", "description": "Compute lowest free energy curve from 2D data.", "syntax": "lowestcurve <dataset> [out <file>] [step <s>]"},
    "meltcurve":          {"category": "Analysis",     "title": "Melting Curve (meltcurve)",              "description": "Generate melting curve from temperature-dependent data.", "syntax": "meltcurve <dataset> [out <file>] [norm]"},
    "multicurve":         {"category": "Analysis",     "title": "Multi-curve Fit (multicurve)",           "description": "Fit multiple exponential curves to data.", "syntax": "multicurve [<dataset>] [out <file>] [nexp <N>]"},
    "multihist":          {"category": "Analysis",     "title": "Multi-histogram (multihist)",            "description": "Histogram multiple data sets simultaneously.", "syntax": "multihist <set1> [<set2>...] [out <file>] [bins <N>]"},
    "calcstate":          {"category": "Analysis",     "title": "Calculate State (calcstate)",            "description": "Calculate state of system using HMM or thresholds.", "syntax": "calcstate [name <name>] [out <file>] <state_args>"},
    "checkoverlap":       {"category": "Analysis",     "title": "Check Overlaps (checkoverlap)",          "description": "Check for bad atomic overlaps/clashes.", "syntax": "check [<mask>] [cut <cut>] [noimage] [out <file>]"},
    "cphstats":           {"category": "Analysis",     "title": "Constant-pH Stats (cphstats)",           "description": "Analyze constant-pH simulation statistics.", "syntax": "cphstats <cpin> {<cpout> [<cpout2> ...]} [out <file>] [deprot <file>]"},
    "remlog":             {"category": "Analysis",     "title": "REMD Log Analysis (remlog)",             "description": "Analyze replica exchange log files.", "syntax": "remlog <remlogfile> [out <file>] [nstlim <N>] [temp0 <T>]"},
    # ── Reference ────────────────────────────────────────────────────────────
    "mask_syntax":        {"category": "Reference",    "title": "Atom Mask Syntax",                       "description": "cpptraj atom selection: :resnum @atomname :resname ! & | < >. Examples: :1-100 @CA !:WAT :LIG<:5.0", "syntax": ":1-100  @CA  !:WAT  :LIG<:5.0  @CA,C,N,O"},
}

SCRIPT_TEMPLATES = {
    "basic_rmsd": {
        "title": "RMSD + RMSF + Rg",
        "description": "Backbone RMSD, per-residue RMSF, radius of gyration",
        "script": "parm topology.prmtop\ntrajin trajectory.nc\n\nautoimage\ncenter !:WAT origin\n\nrmsd backbone @CA,C,N,O first out rmsd.dat\natomicfluct rmsf @CA byres out rmsf.dat\nradgyr rg !:WAT mass out rg.dat\n\ngo\n",
    },
    "full_protein": {
        "title": "Full Protein Analysis",
        "description": "RMSD, RMSF, Rg, H-bonds, secondary structure",
        "script": "parm topology.prmtop\ntrajin trajectory.nc\n\nautoimage\ncenter !:WAT origin\n\nrmsd bb_rmsd @CA,C,N,O first out rmsd.dat\natomicfluct rmsf @CA byres out rmsf.dat\nradgyr rg !:WAT mass out rg.dat\nhbond hbonds !:WAT out hbond.dat avgout hbond_avg.dat\nsecstruct ss out secstruct.dat sumout secstruct_sum.dat\n\ngo\n",
    },
    "clustering": {
        "title": "Trajectory Clustering",
        "description": "Hierarchical clustering + representative structures",
        "script": "parm topology.prmtop\ntrajin trajectory.nc\n\nautoimage\nstrip :WAT,Na+,Cl-\n\ncluster clusters @CA hieragglo epsilon 2.0 sieve 10 out cluster_assign.dat summary cluster_sum.dat info cluster_info.dat repout cluster_rep repfmt pdb\n\ngo\n",
    },
    "pca": {
        "title": "PCA",
        "description": "Covariance matrix + projection onto first 3 modes",
        "script": "parm topology.prmtop\ntrajin trajectory.nc\n\nautoimage\nalign @CA reference\n\nmatrix covar pca_mat @CA out covar.dat\nanalyze modes eigenvalues evectors pca_mat out pca_modes.dat\nprojection pca_proj evecvecs pca_mat @CA out pca_proj.dat beg 1 end 3\n\ngo\n",
    },
    "strip_solvent": {
        "title": "Strip Solvent & Save",
        "description": "Remove water/ions, write protein-only trajectory",
        "script": "parm topology.prmtop\ntrajin trajectory.nc\n\nautoimage\nstrip :WAT,Na+,Cl-\n\ntrajout protein_traj.nc\n\ngo\n",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION + CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

# Section header patterns in the manual, e.g. "11.1 rmsd", "8.3 hbond"
_SECTION_RE = re.compile(
    r'^(\d+\.\d+(?:\.\d+)?)\s+([a-zA-Z][a-zA-Z0-9_\-|/]{1,30})\s*$',
    re.MULTILINE,
)
# Chapter headers like "8 General Commands", "11 Action Commands"
_CHAPTER_RE = re.compile(r'^(\d+)\s+([A-Z][A-Za-z ]{3,50})\s*$', re.MULTILINE)

_MIN_CHUNK_CHARS = 100
_MAX_CHUNK_CHARS = 6000


def _extract_pdf_text() -> list[dict]:
    """
    Extract text from CpptrajManual.pdf and return a list of page dicts:
      [{"page": int, "text": str}, ...]
    Results are cached in cpptraj_manual_cache.json.
    """
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)

    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required to parse the manual: pip install pdfplumber")

    print("[RAG] Extracting text from CpptrajManual.pdf (one-time, ~10 s)…")
    pages = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text})

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False)

    print(f"[RAG] Extracted {len(pages)} pages, cached to {CACHE_PATH.name}")
    return pages


def _chunk_manual(pages: list[dict]) -> list[dict]:
    """
    Split the full manual text into semantic chunks.

    Strategy:
    - Detect section headers (e.g. "11.1 rmsd") as chunk boundaries.
    - Each chunk = one command section (header + body until next header).
    - Also add whole-page chunks for pages that don't fit the pattern.
    - Merge tiny chunks with the previous one.
    """
    # Concatenate all pages preserving page breaks
    full_text = ""
    page_offsets = []  # (start_char, page_num)
    for p in pages:
        page_offsets.append((len(full_text), p["page"]))
        full_text += p["text"] + "\n\n"

    def char_to_page(pos: int) -> int:
        pg = 1
        for start, pnum in page_offsets:
            if start > pos:
                break
            pg = pnum
        return pg

    # Find all section boundaries
    boundaries = []
    for m in _SECTION_RE.finditer(full_text):
        boundaries.append((m.start(), m.group(0).strip(), m.group(2).lower()))
    # Also add chapter boundaries
    for m in _CHAPTER_RE.finditer(full_text):
        boundaries.append((m.start(), m.group(0).strip(), m.group(2).lower()))

    boundaries.sort(key=lambda x: x[0])

    chunks = []
    for i, (pos, header, cmd_name) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
        text = full_text[pos:end].strip()

        if len(text) < _MIN_CHUNK_CHARS:
            continue

        # Trim very long chunks (take first MAX_CHUNK_CHARS)
        if len(text) > _MAX_CHUNK_CHARS:
            text = text[:_MAX_CHUNK_CHARS] + "\n… [truncated]"

        chunks.append({
            "id":       f"manual_sec_{i}",
            "header":   header,
            "cmd_name": cmd_name,
            "text":     text,
            "page":     char_to_page(pos),
            "source":   "CpptrajManual.pdf",
        })

    # Fallback: if very few chunks found, fall back to page-level chunking
    if len(chunks) < 20:
        print("[RAG] Section detection found few chunks — falling back to page-level chunking")
        chunks = []
        for p in pages:
            if len(p["text"]) < _MIN_CHUNK_CHARS:
                continue
            chunks.append({
                "id":       f"page_{p['page']}",
                "header":   f"Page {p['page']}",
                "cmd_name": "",
                "text":     p["text"][:_MAX_CHUNK_CHARS],
                "page":     p["page"],
                "source":   "CpptrajManual.pdf",
            })

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CPPTrajKnowledgeBase:
    """
    RAG over the real CpptrajManual.pdf using TF-IDF retrieval.
    Falls back gracefully if the PDF is not found.
    """

    def __init__(self):
        self._chunks:    list[dict]        = []
        self._texts:     list[str]         = []
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix                  = None
        self._pdf_available                = False

        self._load()

    def _load(self):
        if not PDF_PATH.exists():
            print(f"[RAG] Warning: {PDF_PATH} not found — using built-in command docs only.")
            self._build_fallback_index()
            return

        try:
            pages  = _extract_pdf_text()
            chunks = _chunk_manual(pages)
            if not chunks:
                self._build_fallback_index()
                return

            self._chunks = chunks
            self._texts  = [c["text"] for c in chunks]
            self._pdf_available = True
            print(f"[RAG] Loaded {len(chunks)} chunks from manual (pages 1–{pages[-1]['page']})")
        except Exception as e:
            print(f"[RAG] PDF load error: {e} — using built-in docs.")
            self._build_fallback_index()
            return

        self._build_tfidf()

    def _build_fallback_index(self):
        """Build a minimal TF-IDF index from the built-in CPPTRAJ_COMMANDS."""
        for k, doc in CPPTRAJ_COMMANDS.items():
            text = f"{doc['title']} {doc['description']} {doc['syntax']} {k}"
            self._chunks.append({"id": k, "header": doc["title"], "cmd_name": k,
                                  "text": text, "page": 0, "source": "built-in"})
            self._texts.append(text)
        self._build_tfidf()

    def _build_tfidf(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
            max_features=50_000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self._texts)

    # ── Public API ────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 6) -> list[dict]:
        """Return top-k most relevant chunks for a query."""
        if self.vectorizer is None:
            return []
        q_vec  = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"chunk": self._chunks[i], "score": float(scores[i])}
            for i in top_idx if scores[i] > 0.0
        ]

    def get_context_for_llm(self, query: str, top_k: int = 6) -> str:
        """
        Return a formatted block of the most relevant manual sections
        to inject as context into Claude's prompt.
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant documentation found."

        lines = [
            "=== CPPTRAJ MANUAL — RELEVANT SECTIONS ===",
            f"(Extracted from CpptrajManual.pdf, {len(self._chunks)} total sections indexed)\n",
        ]
        for r in results:
            c = r["chunk"]
            pg = f"p.{c['page']}" if c["page"] else c["source"]
            lines.append(f"--- {c['header']}  [{pg}  relevance:{r['score']:.2f}] ---")
            lines.append(c["text"])
            lines.append("")

        return "\n".join(lines)

    def get_all_commands(self)    -> dict: return CPPTRAJ_COMMANDS
    def get_command(self, key)    -> dict | None: return CPPTRAJ_COMMANDS.get(key)
    def get_categories(self)      -> list: return sorted(set(d["category"] for d in CPPTRAJ_COMMANDS.values()))
    def get_by_category(self, cat)-> dict: return {k: v for k, v in CPPTRAJ_COMMANDS.items() if v["category"] == cat}
    def get_script_templates(self)-> dict: return SCRIPT_TEMPLATES

    @property
    def pdf_available(self) -> bool:
        return self._pdf_available

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)
