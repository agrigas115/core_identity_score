#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch core identity scoring.
Processes all .pdb files in a user-specified directory.

@author: atgrigas
"""
import argparse
import sys
import os
import glob
import time

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
    structure = freesasa.Structure()
    radii = []
    for aname, rname, res_id, chain, x, y, z in atoms_data:
        r = sigma.get((rname, aname), 0.0)
        radii.append(r)
        structure.addAtom(aname, rname, str(res_id), chain, x, y, z)
    structure.setRadii(radii)
    return structure, radii


def get_residue_atoms(topology, positions):
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
    residues = get_residue_atoms(topology, positions)
    n_res = len(residues)

    prev_atoms = {'CA', 'C', 'O'}
    next_atoms = {'N', 'H', 'CA'}

    ref_sasa = {}
    for i, res in enumerate(residues):
        fragment = []
        central_start = 0

        if i > 0:
            prev = residues[i - 1]
            for aname, x, y, z in prev['atoms']:
                if aname in prev_atoms:
                    fragment.append((aname, prev['res_name'], prev['res_index'], prev['chain'], x, y, z))
            central_start = len(fragment)

        for aname, x, y, z in res['atoms']:
            fragment.append((aname, res['res_name'], res['res_index'], res['chain'], x, y, z))
        central_end = len(fragment)

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
    context_sasa = compute_sasa(topology, positions, sigma)
    reference_sasa = compute_reference_sasa(topology, positions, sigma)

    rsasa = {}
    for res in topology.residues():
        ri = res.index
        ref = reference_sasa.get(ri, 0.0)
        ctx = context_sasa.get(ri, 0.0)
        if ref > 0:
            rsasa[ri] = (res.name, res.id, ctx / ref)
        else:
            rsasa[ri] = (res.name, res.id, 0.0)

    return rsasa


# ---------------------------------------------------------------
# Sequence and label extraction
# ---------------------------------------------------------------

def extract_sequence_and_labels(rsasa, threshold=0.1):
    sequence = []
    rsasa_values = []
    core_labels = []
    res_ids = []

    for i in sorted(rsasa.keys()):
        name, res_id, val = rsasa[i]
        one = THREE_TO_ONE.get(name, 'X')
        sequence.append(one)
        rsasa_values.append(val)
        core_labels.append(1 if val < threshold else 0)
        res_ids.append(res_id)

    return ''.join(sequence), rsasa_values, core_labels, res_ids


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
# ESM2 embedding extraction (accepts preloaded model)
# ---------------------------------------------------------------

def load_esm2(model_name="esm2_t33_650M_UR50D"):
    """Load ESM2 model and alphabet once."""
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    return model, alphabet


def get_esm2_embeddings(sequence, esm_model, alphabet, layer=33):
    """
    Extract per-residue embeddings from a preloaded ESM2 model.
    """
    batch_converter = alphabet.get_batch_converter()

    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[layer], return_contacts=False)

    embeddings = results["representations"][layer][0, 1:len(sequence)+1, :]

    return embeddings.cpu()


# ---------------------------------------------------------------
# Inference (accepts preloaded classifier)
# ---------------------------------------------------------------

def load_classifier(model_path, device=None):
    """Load the SASAClassifier once."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SASAClassifier(input_dim=1280)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device


def predict_core(embeddings, classifier, device):
    """
    Run pretrained SASAClassifier on ESM2 embeddings.
    """
    with torch.no_grad():
        logits = classifier(embeddings.float().to(device))
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)

    pred_labels = (1 - preds).cpu().numpy()
    pred_scores = (1 - probs).cpu().numpy()

    return pred_labels, pred_scores


def write_output(filepath, sequence, rsasa_values, core_labels, pred_labels, mcc_score, lddt_pred, res_ids):
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
        lines.append(f"{res_ids[i]:<6}{sequence[i]:<4}{rsasa_values[i]:<8.4f}{t:<6}{p:<6}{m:<6}")

    text = "\n".join(lines)

    with open(filepath, 'w') as f:
        f.write(text + "\n")


# ---------------------------------------------------------------
# Main: batch processing
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch core identity scoring: predict fold quality for all PDB files in a directory"
    )
    parser.add_argument("input_dir", help="Directory containing PDB files")
    parser.add_argument("--pH", type=float, default=7.0, help="pH for protonation states (default: 7.0)")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for output files (default: same as input_dir)")
    parser.add_argument("--summary", default="batch_summary.csv",
                        help="Summary CSV filename (written to output_dir, default: batch_summary.csv)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"Error: directory not found: {args.input_dir}")

    output_dir = args.output_dir if args.output_dir else args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    pdb_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pdb")))
    if not pdb_files:
        sys.exit(f"No .pdb files found in {args.input_dir}")

    print(f"Found {len(pdb_files)} PDB files in {args.input_dir}")

    # --- Load shared resources once ---
    freesasa.setVerbosity(freesasa.nowarnings)
    sigma = parse_sigma_file('vdw_jennifer_dict.txt')

    print("Loading ESM2 model...")
    esm_model, alphabet = load_esm2()

    MODEL_PATH = "model_weights/best_sasa_multilayer.pt"
    print("Loading classifier...")
    classifier, device = load_classifier(MODEL_PATH)

    coeffs = [-0.22988289, 0.90041124, 0.11350768, 0.26786789]
    fit_fn = np.poly1d(coeffs)

    # --- Process each structure ---
    summary_rows = []

    for idx, pdb_path in enumerate(pdb_files):
        basename = os.path.basename(pdb_path)
        name = os.path.splitext(basename)[0]
        print(f"\n[{idx+1}/{len(pdb_files)}] {basename}")

        t0 = time.time()

        try:
            print("  Adding hydrogens...")
            topology, positions = protonate(pdb_path, pH=args.pH)

            print("  Measuring rSASA...")
            rsasa = compute_rsasa(topology, positions, sigma)
            sequence, rsasa_values, core_labels, res_ids = extract_sequence_and_labels(rsasa)

            print("  Getting ESM2 embeddings...")
            embeddings = get_esm2_embeddings(sequence, esm_model, alphabet)

            print("  Predicting core...")
            pred_labels, pred_scores = predict_core(embeddings, classifier, device)

            mcc_score = mcc(pred_labels, core_labels)
            lddt_pred = fit_fn(mcc_score)

            elapsed = time.time() - t0
            print(f"  Predicted LDDT: {lddt_pred:.3f}  (MCC={mcc_score:.3f}, {elapsed:.1f}s)")

            output_file = os.path.join(output_dir, name + "_core_identity_score.txt")
            write_output(output_file, sequence, rsasa_values, core_labels,
                         pred_labels, mcc_score, lddt_pred, res_ids)
            print(f"  Wrote {output_file}")

            summary_rows.append({
                'file': basename,
                'L': len(sequence),
                'MCC': mcc_score,
                'LDDT_pred': lddt_pred,
                'n_core_true': sum(core_labels),
                'n_core_pred': int(sum(pred_labels)),
                'status': 'OK',
            })

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED ({elapsed:.1f}s): {e}")
            summary_rows.append({
                'file': basename,
                'L': 0,
                'MCC': float('nan'),
                'LDDT_pred': float('nan'),
                'n_core_true': 0,
                'n_core_pred': 0,
                'status': str(e),
            })

    # --- Write summary CSV ---
    summary_path = os.path.join(output_dir, args.summary)
    with open(summary_path, 'w') as f:
        f.write("file,L,MCC,LDDT_pred,n_core_true,n_core_pred,status\n")
        for row in summary_rows:
            f.write(f"{row['file']},{row['L']},{row['MCC']:.4f},{row['LDDT_pred']:.4f},"
                    f"{row['n_core_true']},{row['n_core_pred']},{row['status']}\n")

    print(f"\n{'='*50}")
    print(f"Batch complete: {sum(1 for r in summary_rows if r['status']=='OK')}/{len(summary_rows)} succeeded")
    print(f"Summary written to {summary_path}")