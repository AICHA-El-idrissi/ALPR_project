#!/usr/bin/env python3
"""
Inspecte un modèle ONNX AVANT de tenter sa conversion vers ncnn.


Ce que le script affiche :
  - la présence (ou non) de données externes (fichier .onnx.data)
  - la version d'opset
  - les entrées et sorties, avec leurs shapes et les dimensions dynamiques
  - le décompte des opérateurs
  - la liste des opérateurs connus pour poser problème à onnx2ncnn

Usage :
    python scripts/probe_onnx.py models/dfine_n/best_stg2.onnx
    python scripts/probe_onnx.py models/paddleocr/paddleocr_rec.onnx
    python scripts/probe_onnx.py models/dfine_n/best_stg2.onnx \
        --inline models/dfine_n/best_stg2_single.onnx

L'option --inline réécrit le modèle en UN SEUL fichier, poids inclus :
onnx2ncnn lit un protobuf unique et ne sait pas résoudre un `.onnx.data`
voisin. Sans cette étape, la conversion d'un modèle à données externes
échoue ou produit un modèle aux poids vides.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import onnx

# Opérateurs qui font échouer onnx2ncnn, ou qu'il convertit en produisant un
# modèle qui ne donnera pas les mêmes sorties. La valeur est la raison, telle
# qu'elle sera affichée. Ce n'est pas une liste exhaustive : c'est celle des
# cas rencontrés sur les familles de modèles qui nous concernent (DETR et
# dérivés, OCR séquentiel).
OPS_A_RISQUE = {
    "GridSample":         "attention déformable (DETR) -- support ncnn partiel selon la version",
    "NonMaxSuppression":  "NMS intégrée au graphe -- à exporter SANS NMS",
    "RoiAlign":           "non géré par onnx2ncnn",
    "Einsum":             "non géré par onnx2ncnn",
    "ScatterND":          "non géré par onnx2ncnn",
    "NonZero":            "sortie de taille dynamique -- incompatible avec ncnn",
    "TopK":               "souvent utilisé pour la sélection des requêtes DETR",
    "If":                 "sous-graphe conditionnel -- non géré",
    "Loop":               "sous-graphe itératif -- non géré",
    "LSTM":               "tête séquentielle CRNN -- vérifier le support ncnn",
    "GRU":                "tête séquentielle -- vérifier le support ncnn",
    "Range":              "génère une dimension dynamique",
    "LayerNormalization": "support selon la version de ncnn (sinon décomposer)",
    "MultiHeadAttention": "non géré -- à décomposer",
    "Pad":                "padding dynamique non géré (padding constant OK)",
}


def _shape(tensor) -> str:
    """Shape lisible, avec les dimensions dynamiques mises en évidence."""
    dims = []
    for d in tensor.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(str(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(f"<{d.dim_param}>")   # dimension dynamique nommée
        else:
            dims.append("<?>")
    return "[" + ", ".join(dims) + "]"


def _type(tensor) -> str:
    return onnx.TensorProto.DataType.Name(tensor.type.tensor_type.elem_type)


def inspecter(chemin: Path, inline: Path | None) -> int:
    if not chemin.exists():
        print(f"Fichier introuvable : {chemin}")
        return 1

    data_file = Path(str(chemin) + ".data")
    taille = chemin.stat().st_size / 1e6
    print(f"\n=== {chemin} ===")
    print(f"  taille du .onnx     : {taille:.1f} Mo")
    if data_file.exists():
        print(f"  données externes    : {data_file.name} "
              f"({data_file.stat().st_size / 1e6:.1f} Mo)")
        print("    [!] onnx2ncnn ne résout PAS les données externes : "
              "utilise --inline avant de convertir.")
    else:
        print("  données externes    : aucune (modèle en un seul fichier)")

    # load_external_data=True : indispensable, sinon les poids restent vides
    # et tout ce qui suit (y compris un éventuel --inline) serait faux.
    modele = onnx.load(str(chemin), load_external_data=True)

    opsets = {imp.domain or "ai.onnx": imp.version for imp in modele.opset_import}
    print(f"  opset               : {opsets}")
    print(f"  producteur          : {modele.producer_name or '?'} "
          f"{modele.producer_version or ''}")

    print("\n  Entrées :")
    initialiseurs = {i.name for i in modele.graph.initializer}
    for e in modele.graph.input:
        if e.name in initialiseurs:      # poids déclarés comme entrée : ignorés
            continue
        print(f"    - {e.name:<24} {_type(e):<8} {_shape(e)}")

    print("  Sorties :")
    for s in modele.graph.output:
        print(f"    - {s.name:<24} {_type(s):<8} {_shape(s)}")

    compte = Counter(n.op_type for n in modele.graph.node)
    print(f"\n  Opérateurs : {sum(compte.values())} nœuds, "
          f"{len(compte)} types distincts")
    for op, n in compte.most_common(12):
        print(f"    {n:>5}  {op}")
    if len(compte) > 12:
        print(f"    ...   ({len(compte) - 12} autres types)")

    risques = {op: OPS_A_RISQUE[op] for op in compte if op in OPS_A_RISQUE}
    if risques:
        print("\n  [!] Opérateurs à risque pour onnx2ncnn :")
        for op, raison in risques.items():
            print(f"    - {op} (x{compte[op]}) : {raison}")
        print("    -> privilégier `pnnx` (plus tolérant qu'onnx2ncnn), "
              "et vérifier les sorties après conversion.")
    else:
        print("\n  Aucun opérateur de la liste à risque détecté.")

    if inline is not None:
        inline.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(modele, str(inline), save_as_external_data=False)
        print(f"\n  Modèle réécrit en un seul fichier : {inline} "
              f"({inline.stat().st_size / 1e6:.1f} Mo)")
        print("  C'est CE fichier qu'il faut donner à onnx2ncnn / pnnx.")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modele", type=Path, help="chemin du fichier .onnx")
    ap.add_argument("--inline", type=Path, default=None,
                    help="réécrit le modèle en un seul fichier (poids inclus)")
    args = ap.parse_args()
    return inspecter(args.modele, args.inline)


if __name__ == "__main__":
    sys.exit(main())