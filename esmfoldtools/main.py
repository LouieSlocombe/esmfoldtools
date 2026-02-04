import numpy as np
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from pdbfixer import PDBFixer
from openmm import app


def fix_pdb(file_in, file_out, ph=7.0):
    """
    Fixes and prepares a PDB file for molecular simulations.

    This function uses the `PDBFixer` class to process a PDB file by performing
    the following tasks:
    - Identifies and adds missing residues.
    - Replaces nonstandard residues with standard ones.
    - Removes heterogens (e.g., ligands, ions, water molecules).
    - Identifies and adds missing atoms.
    - Adds missing hydrogens based on the specified pH.

    Parameters
    ----------
    file_in : str
        The input file path of the PDB file to be fixed.
    file_out : str
        The output file path where the fixed PDB file will be saved.
    ph : float, optional
        The pH value used to add missing hydrogens. Default is 7.0.

    Returns
    -------
    None
        The function does not return any value. The fixed PDB file is written
        to the specified output file.

    Notes
    -----
    - The function uses the `PDBFixer` class from the `pdbfixer` library.
    - The output PDB file is written using the `openmm.app.PDBFile` class.
    """
    fixer = PDBFixer(filename=file_in)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    app.PDBFile.writeFile(fixer.topology, fixer.positions, open(file_out, 'w'))
    return None


def fold_sequence_esm(sequence: str, fileout: str | None = None) -> tuple[str, np.ndarray, float]:
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    model.esm = model.esm.half()
    torch.backends.cuda.matmul.allow_tf32 = True
    model.trunk.set_chunk_size(64)

    inputs = tok([sequence],
                 return_tensors="pt",
                 add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == device):
        out = model(**inputs)
        pdb_str = model.infer_pdb(sequence)

    # Write the folded structure to the output PDB file
    if fileout:
        with open(fileout, "w") as f:
            f.write(pdb_str)

    # Extract pLDDT (per-residue confidence) and pTM (predicted TM-score)
    plddt = out.plddt[0].cpu().numpy()
    ptm = out.ptm.cpu().numpy()

    return pdb_str, plddt, ptm
