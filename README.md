[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agrigas115/core_identity_score/blob/main/core_identity_score.ipynb)
# Core Identity Score

Predict protein fold quality from sequence alone using binary core/surface identity.

Given a protein structure (PDB file), this tool:
1. Adds hydrogens using PDBFixer
2. Computes per-residue relative solvent accessible surface area (rSASA) using FreeSASA
3. Assigns binary core (rSASA < 0.1) and surface labels
4. Extracts per-residue ESM2 embeddings from the sequence
5. Predicts core/surface labels using a trained classification head
6. Compares predicted and true labels via MCC and reports an estimated LDDT

The key finding is that binary core identity is a near-minimal sufficient statistic for fold quality, achieving ρ ≈ 0.94 correlation with LDDT when the prediction of core identity is exact. Using the ESM2 prediction, the correlation reaches ρ ≈ 0.82.

## Installation

Requires Python ≥ 3.9. Install dependencies with conda:

```
conda create -n core-score -c conda-forge python=3.9 pdbfixer biopython freesasa pytorch numpy
conda activate core-score
pip install fair-esm
```

The ESM2 model weights (~2.5 GB) are downloaded automatically on first run.

## Usage

```
python main.py structure.pdb
```

Options:
- `--pH` : pH for protonation states (default: 7.0)

Output is written to `<input>_core_identity_score.txt`.

An example decoy from CASP is included: T0762TS008_1_H.pdb
With ESM2 previously downloaded, this example runs in 8 seconds.

## Output format

```
# LDDT(pred)=0.7812  MCC=0.8234  L=150
# True core: 45  Pred core: 43
# TP=40  FP=3  FN=5  TN=102
# C=core (rSASA<0.1)  S=surface

Idx   AA  rSASA   True  Pred  Match
------------------------------------
1     M   0.3200  S     S     =
2     E   0.0500  C     C     =
3     T   0.4100  S     S     =
4     L   0.0200  C     S     X
...
```

## Files

- `main.py` — main script
- `vdw_jennifer_dict.txt` — custom hard-sphere atomic radii from [Gaines et al. PRE 2016](https://doi.org/10.1103/PhysRevE.93.032415)
- `model_weights/best_sasa_multilayer.pt` — trained ESM2 classification head

## Method

Atomic radii are from the explicit hydrogen hard-sphere model validated against side-chain dihedral angle distributions in protein crystal structures (Gaines et al. 2016). rSASA is computed per-residue as the ratio of SASA in the full protein context to SASA in an isolated dipeptide mimetic (Cα-C-O of preceding residue, full central residue, N-H-Cα of following residue), using the Lee-Richards algorithm with a 1.4 Å probe radius.

ESM2 (650M parameter model, layer 33) provides per-residue embeddings that are passed through a trained two-layer classification head to predict core/surface identity. The MCC between predicted and true labels is converted to an estimated LDDT via a cubic polynomial fit calibrated on CASP decoy structures.

## Citation

If you use this tool, please cite:
