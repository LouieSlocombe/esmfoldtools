import os

import esmfoldtools as ft


def test_esmfold_folding():
    print(flush=True)
    # https://huggingface.co/docs/transformers/model_doc/esm#transformers.EsmForProteinFolding

    seq = "MGSDKIHHHHHHENLYFQG"
    ft.fold_sequence_esm(seq, fileout="tmp.pdb")

    # Assert that the pdb file was created
    assert os.path.isfile("tmp.pdb")
    os.remove("tmp.pdb")
