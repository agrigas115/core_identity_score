#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:35:10 2026

@author: atgrigas
"""
import argparse
import sys
import os
os.environ["TORCH_HOME"] = os.path.expanduser("~/.my_cache/torch")

from pdbfixer import PDBFixer
from openmm.app import PDBFile
import freesasa
import torch
import torch.nn as nn
import esm
import numpy as np


# ---------------------------------------------------------------
# Three-letter to one-letter mapping
# ---------------------------------------------------------------

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'ASX': 'B', 'GLX': 'Z',
}


def mcc(true_labels, pred_labels):
    """Binary Matthews correlation coefficient."""
    tp = sum(t == 1 and p == 1 for t, p in zip(true_labels, pred_labels))
    tn = sum(t == 0 and p == 0 for t, p in zip(true_labels, pred_labels))
    fp = sum(t == 0 and p == 1 for t, p in zip(true_labels, pred_labels))
    fn = sum(t == 1 and p == 0 for t, p in zip(true_labels, pred_labels))

    num = tp * tn - fp * fn
    den = ((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)) ** 0.5

    return num / den if den > 0 else 0.0

# ---------------------------------------------------------------
# Load and protonate
# ---------------------------------------------------------------

def protonate(input_path, pH=7.0):
    """Load a PDB file, add missing hydrogens. Returns (topology, positions)."""

    if not os.path.isfile(input_path):
        sys.exit(f"Error: file not found: {input_path}")

    fixer = PDBFixer(filename=input_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=pH)

    return fixer.topology, fixer.positions


# ---------------------------------------------------------------
# Sigma file parsing
# ---------------------------------------------------------------

def parse_sigma_file(path):
    """
    Parse a residue/atom radius file.
    Returns dict: (residue_name, atom_name) -> radius in Angstroms.
    """
    sigma = {}
    current_res = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("RESIDUE"):
                parts = line.split()
                current_res = parts[2]
            elif line.startswith("ATOM"):
                parts = line.split()
                atom_name = parts[1]
                radius = float(parts[2])
                sigma[(current_res, atom_name)] = radius

    return sigma


# ---------------------------------------------------------------
# FreeSASA helpers
# ---------------------------------------------------------------


SASA_PARAMS = freesasa.Parameters({
    'algorithm': freesasa.LeeRichards,
    'probe-radius': 1.4,
    'n-slices': 200,
})


def build_freesasa_structure(atoms_data, sigma):
    """
    Build a freesasa.Structure from a list of (atom_name, res_name, res_id, chain, x, y, z).
    Returns (structure, radii).
    """
    structure = freesasa.Structure()
    radii = []
    for aname, rname, res_id, chain, x, y, z in atoms_data:
        r = sigma.get((rname, aname), 0.0)
        radii.append(r)
        structure.addAtom(aname, rname, str(res_id), chain, x, y, z)
    structure.setRadii(radii)
    return structure, radii


def get_residue_atoms(topology, positions):
    """
    Organize atoms by residue. Returns list of dicts, one per residue:
        {
            'res_name': str,
            'res_index': int,
            'chain': str,
            'atoms': [(atom_name, x, y, z), ...]
        }
    Coordinates converted to Angstroms.
    """
    residues = []
    for res in topology.residues():
        atoms = []
        for atom in res.atoms():
            pos = positions[atom.index]
            x = pos.x * 10.0
            y = pos.y * 10.0
            z = pos.z * 10.0
            atoms.append((atom.name, x, y, z))
        residues.append({
            'res_name': res.name,
            'res_index': res.index,
            'chain': res.chain.id,
            'atoms': atoms,
        })
    return residues


# ---------------------------------------------------------------
# SASA computation
# ---------------------------------------------------------------

def compute_sasa(topology, positions, sigma):
    """
    Compute per-residue SASA in the full protein context.
    Returns dict: res_index -> SASA (Angstrom^2), summed over that residue's atoms.
    """
    residues = get_residue_atoms(topology, positions)

    all_atoms = []
    atom_to_res = []
    for res in residues:
        for aname, x, y, z in res['atoms']:
            all_atoms.append((aname, res['res_name'], res['res_index'], res['chain'], x, y, z))
            atom_to_res.append(res['res_index'])

    structure, radii = build_freesasa_structure(all_atoms, sigma)
    result = freesasa.calc(structure, SASA_PARAMS)

    res_sasa = {}
    for i in range(structure.nAtoms()):
        ri = atom_to_res[i]
        res_sasa[ri] = res_sasa.get(ri, 0.0) + result.atomArea(i)

    return res_sasa


def compute_reference_sasa(topology, positions, sigma):
    """
    Compute per-residue reference SASA from dipeptide mimetics.
    For residue i, the mimetic is:
        - previous residue: CA, C, O
        - current residue: all atoms
        - next residue: N, H, CA
    SASA is summed only over the central residue's atoms.
    Returns dict: res_index -> reference SASA (Angstrom^2).
    """
    residues = get_residue_atoms(topology, positions)
    n_res = len(residues)

    prev_atoms = {'CA', 'C', 'O'}
    next_atoms = {'N', 'H', 'CA'}

    ref_sasa = {}
    for i, res in enumerate(residues):
        fragment = []
        central_start = 0

        # previous residue partial
        if i > 0:
            prev = residues[i - 1]
            for aname, x, y, z in prev['atoms']:
                if aname in prev_atoms:
                    fragment.append((aname, prev['res_name'], prev['res_index'], prev['chain'], x, y, z))
            central_start = len(fragment)

        # current residue: all atoms
        for aname, x, y, z in res['atoms']:
            fragment.append((aname, res['res_name'], res['res_index'], res['chain'], x, y, z))
        central_end = len(fragment)

        # next residue partial
        if i < n_res - 1:
            nxt = residues[i + 1]
            for aname, x, y, z in nxt['atoms']:
                if aname in next_atoms:
                    fragment.append((aname, nxt['res_name'], nxt['res_index'], nxt['chain'], x, y, z))

        structure, radii = build_freesasa_structure(fragment, sigma)
        result = freesasa.calc(structure, SASA_PARAMS)

        sasa = 0.0
        for j in range(central_start, central_end):
            sasa += result.atomArea(j)

        ref_sasa[res['res_index']] = sasa

    return ref_sasa


def compute_rsasa(topology, positions, sigma):
    """
    Compute rSASA for each residue: SASA_in_context / SASA_as_dipeptide.
    Returns dict: res_index -> (res_name, rSASA).
    """
    context_sasa = compute_sasa(topology, positions, sigma)
    reference_sasa = compute_reference_sasa(topology, positions, sigma)

    rsasa = {}
    for res in topology.residues():
        ri = res.index
        ref = reference_sasa.get(ri, 0.0)
        ctx = context_sasa.get(ri, 0.0)
        if ref > 0:
            rsasa[ri] = (res.name, ctx / ref)
        else:
            rsasa[ri] = (res.name, 0.0)

    return rsasa


# ---------------------------------------------------------------
# Sequence and label extraction
# ---------------------------------------------------------------

def extract_sequence_and_labels(rsasa, threshold=0.1):
    """
    From the rsasa dict {res_index: (res_name, rSASA)},
    extract the one-letter sequence, rSASA values, and binary core labels.
    Returns (sequence, rsasa_values, core_labels) where:
        sequence: str of one-letter codes
        rsasa_values: list of floats
        core_labels: list of ints (1 = core, 0 = surface)
    """
    sequence = []
    rsasa_values = []
    core_labels = []

    for i in sorted(rsasa.keys()):
        name, val = rsasa[i]
        one = THREE_TO_ONE.get(name, 'X')
        sequence.append(one)
        rsasa_values.append(val)
        core_labels.append(1 if val < threshold else 0)

    return ''.join(sequence), rsasa_values, core_labels


# ---------------------------------------------------------------
# Model architecture (must match training)
# ---------------------------------------------------------------

class SASAClassifier(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=256, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------
# ESM2 embedding extraction
# ---------------------------------------------------------------

def get_esm2_embeddings(sequence, model_name="esm2_t33_650M_UR50D", layer=33):
    """
    Extract per-residue embeddings from ESM2.

    Args:
        sequence: one-letter amino acid sequence string
        model_name: ESM2 model identifier
        layer: which transformer layer to extract representations from

    Returns:
        torch tensor of shape (L, d_model)
    """
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer], return_contacts=False)

    # shape: (1, L+2, d_model) — strip BOS and EOS tokens
    embeddings = results["representations"][layer][0, 1:len(sequence)+1, :]

    return embeddings.cpu()


# ---------------------------------------------------------------
# Inference
# ---------------------------------------------------------------

def predict_core(embeddings, model_path, device=None):
    """
    Run trained SASAClassifier on ESM2 embeddings.

    Args:
        embeddings: torch tensor (L, d_model)
        model_path: path to trained model weights
        device: torch device (default: auto)

    Returns:
        pred_labels: numpy array of ints (1 = core, 0 = surface)
        pred_scores: numpy array of floats (P(surface), i.e. class 1 score)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = embeddings.shape[1]
    model = SASAClassifier(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(embeddings.float().to(device))
        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(surface)
        preds = logits.argmax(dim=-1)                 # 0=buried, 1=surface

    # flip convention to match core_labels: 1=core, 0=surface
    pred_labels = (1 - preds).cpu().numpy()
    pred_scores = (1 - probs).cpu().numpy()  # P(core)

    return pred_labels, pred_scores

def write_output(filepath, sequence, rsasa_values, core_labels, pred_labels, mcc_score, lddt_pred):
    """
    Write column-format output comparing true and predicted core labels.

    C = core (rSASA < 0.1), S = surface, = = agree, X = disagree
    """
    L = len(sequence)
    label_char = lambda v: "C" if v == 1 else "S"

    tp = sum(t == 1 and p == 1 for t, p in zip(core_labels, pred_labels))
    fp = sum(t == 0 and p == 1 for t, p in zip(core_labels, pred_labels))
    fn = sum(t == 1 and p == 0 for t, p in zip(core_labels, pred_labels))
    tn = sum(t == 0 and p == 0 for t, p in zip(core_labels, pred_labels))
    n_core_true = sum(core_labels)
    n_core_pred = sum(pred_labels)

    lines = []
    lines.append(f"# LDDT(pred)={lddt_pred:.4f}  MCC={mcc_score:.4f}  L={L}")
    lines.append(f"# True core: {n_core_true}  Pred core: {n_core_pred}")
    lines.append(f"# TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    lines.append(f"# C=core (rSASA<0.1)  S=surface")
    lines.append("")
    lines.append(f"{'Idx':<6}{'AA':<4}{'rSASA':<8}{'True':<6}{'Pred':<6}{'Match':<6}")
    lines.append("-" * 36)

    for i in range(L):
        t = label_char(core_labels[i])
        p = label_char(pred_labels[i])
        m = "=" if core_labels[i] == pred_labels[i] else "X"
        lines.append(f"{i+1:<6}{sequence[i]:<4}{rsasa_values[i]:<8.4f}{t:<6}{p:<6}{m:<6}")

    text = "\n".join(lines)

    with open(filepath, 'w') as f:
        f.write(text + "\n")

    print(f"Wrote {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Core identity score: predict fold quality from ESM2 embeddings")
    parser.add_argument("input", help="Path to input PDB file")
    parser.add_argument("--pH", type=float, default=7.0, help="pH for protonation states (default: 7.0)")
    args = parser.parse_args()
    
    freesasa.setVerbosity(freesasa.nowarnings)
    sigma = parse_sigma_file('vdw_jennifer_dict.txt')
    
    ### Load the structure and add Hydrogens using PDBFixer ###
    print('Adding Hydrogens...')
    file = args.input
    output_file = file.split('.pdb')[0]+'_core_identity_score.txt'
    topology, positions = protonate(file)

    ### Calculate rSASA ###
    print('Measuring rSASA...')
    rsasa = compute_rsasa(topology, positions, sigma)
    sequence, rsasa_values, core_labels = extract_sequence_and_labels(rsasa)
    
    ## Embed sequence into ESM2 ###
    print('Getting ESM2 embeddings...')
    embeddings = get_esm2_embeddings(sequence)
    
    ## Predict core ###
    print('Predicting core...')
    MODEL_PATH = "model_weights/best_sasa_multilayer.pt"
    pred_labels, pred_scores = predict_core(embeddings, MODEL_PATH)
    
    mcc_score = mcc(pred_labels,core_labels)
    
    coeffs = [-0.22988289,  0.90041124,  0.11350768,  0.26786789]
    
    fit_fn = np.poly1d(coeffs)
    lddt_pred = fit_fn(mcc_score)
    
    print('Predicted LDDT: '+str(round(lddt_pred,3)))
    
    write_output(output_file, sequence, rsasa_values, core_labels, pred_labels, mcc_score, lddt_pred)
