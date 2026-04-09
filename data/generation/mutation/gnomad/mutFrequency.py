import sys
import os
sys.path.append(os.getcwd())

import requests
import pandas as pd
import re
import time

from utils.functions import load_yaml
from data.reference.residue_dictionary import residue3to1 as AA3TO1
from data.reference.residue_dictionary import residue1to3 as AA1TO3

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
GNOMAD_API = "https://gnomad.broadinstitute.org/api"

# ──────────────────────────────────────────────
# 1) UniProt → Canonical ENST mapping (MANE Select priority)
# ──────────────────────────────────────────────
def get_canonical_enst(uniprot_id: str) -> str:
    """Fetch canonical Ensembl transcript from UniProt API (MANE Select priority)"""
    r = requests.get(
        f"https://rest.uniprot.org/uniprotkb/{uniprot_id}",
        headers={"Accept": "application/json"}
    )
    data = r.json()

    ensembl_transcripts = []
    mane_select = None

    for xref in data.get("uniProtKBCrossReferences", []):
        if xref["database"] == "MANE-Select":
            mane_select = xref["id"]
        elif xref["database"] == "Ensembl":
            ensembl_transcripts.append(xref["id"])

    enst = mane_select or (ensembl_transcripts[0] if ensembl_transcripts else None)

    # Remove version number: ENST00000357654.9 → ENST00000357654
    if enst and "." in enst:
        enst = enst.split(".")[0]

    return enst

# ──────────────────────────────────────────────
# 2) Fetch variants from gnomAD GraphQL API
# ──────────────────────────────────────────────
def fetch_variants_by_transcript(enst_id: str) -> tuple:
    """Fetch gnomAD v4 missense variants + gene symbol by ENST ID"""
    query = """
        {
        transcript(transcript_id: "%s", reference_genome: GRCh38) {
            transcript_id
            gene { symbol }
            variants(dataset: gnomad_r4) {
            variant_id
            consequence
            hgvsp
            exome { ac an }
            genome { ac an }
            }
        }
        }
        """ % enst_id

    r = requests.post(GNOMAD_API, json={"query": query})
    data = r.json()

    if "errors" in data:
        raise Exception(f"gnomAD API error: {data['errors'][0]['message']}")

    transcript = data.get("data", {}).get("transcript")
    if transcript is None:
        raise Exception(f"Transcript {enst_id} not found in gnomAD")
    
    time.sleep(1)

    return transcript["variants"], transcript["gene"]["symbol"]

# ──────────────────────────────────────────────
# 3) Parse HGVSp
# ──────────────────────────────────────────────
def parse_hgvsp(hgvsp: str):
    """Convert p.Arg123His → ('R', 123, 'H')"""
    if not hgvsp:
        return None, None, None
    match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', hgvsp)
    if match:
        ref_aa = AA3TO1.get(match.group(1), match.group(1))
        position = int(match.group(2))
        alt_aa = AA3TO1.get(match.group(3), match.group(3))
        return ref_aa, position, alt_aa
    return None, None, None

# ──────────────────────────────────────────────
# 4) Full pipeline
# ──────────────────────────────────────────────
def build_dataset(uniprot_ids: list) -> pd.DataFrame:
    """UniProt ID list → residue-level mutation count dataframe"""

    # Step 1: UniProt → canonical ENST mapping
    mapping = {}
    for uid in uniprot_ids:
        enst = get_canonical_enst(uid)
        if enst:
            mapping[uid] = enst
            print(f"[Mapping OK] {uid} → {enst}")
        else:
            print(f"[Mapping FAIL] {uid} → skip")
        time.sleep(0.5)

    # Step 2: Collect variants from gnomAD
    all_records = []

    for uid, enst in mapping.items():
        print(f"[gnomAD query] {uid} ({enst})")

        try:
            variants, gene_symbol = fetch_variants_by_transcript(enst)
        except Exception as e:
            print(f"  → Error: {e}")
            continue

        for v in variants:
            if v.get("consequence") != "missense_variant":
                continue

            ref_aa, pos, alt_aa = parse_hgvsp(v.get("hgvsp"))
            if pos is None:
                continue

            ac = 0
            an = 0
            
            if v.get("exome"):
                ac += v["exome"].get("ac") or 0
                an += v["exome"].get("an") or 0
            
            if v.get("genome"):
                ac += v["genome"].get("ac") or 0
                an += v["genome"].get("an") or 0

            af = ac / an if an > 0 else None

            all_records.append({
                "uniprot_id": uid,
                "enst_id": enst,
                "gene_symbol": gene_symbol,
                "residuetype": AA1TO3[AA3TO1[ref_aa.upper()]].lower(),
                "position": pos,
                "alt_aa": AA1TO3[AA3TO1[alt_aa.upper()]].lower(),
                "allele_count": ac,
                "allele_number": an,
                "allele_frequency": af
            })

        time.sleep(5)  # Rate limit prevention

    # Step 3: Aggregate
    df = pd.DataFrame(all_records)
    if df.empty:
        print("No variants collected.")
        return df
ㅂ
    variant_dict = (
            df.groupby(["uniprot_id", "enst_id", "gene_symbol", "residuetype", "position"])
            .apply(lambda g: g.groupby("alt_aa")["allele_count"].sum().to_dict())
            .reset_index(name="variant")
        )


    # Add mutation_count and total_allele_count
    counts = (
        df.groupby(["uniprot_id", "enst_id", "gene_symbol", "residuetype", "position"])
        .agg(
            unique_mutation_type=("alt_aa", "nunique"),
            total_mutations_count=("allele_count", "sum"),
            total_number=("allele_number", "max") 
        )
        .reset_index()
    )

    counts['frequency'] = counts['total_mutations_count'] / counts['total_number']

    result = counts.merge(variant_dict, on=["uniprot_id", "enst_id", "gene_symbol",
                                              "residuetype", "position"])
    result = (
        result.sort_values(["uniprot_id", "position"])
        .reset_index(drop=True)
    )

    print(f"\nDone: {len(result)} residue positions collected")
    return result

# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":

    config = load_yaml("config/RRGCL.yaml")
    DATABASE = config.DATABASE
    proc_edge_df = pd.read_csv(f"{DATABASE}/step5_rmStrangeEdge_exceptCDSFilter_human_aa_edges_exclubq_Nucleosome_related_data_v031626.csv")

    uniprot_ids = set(proc_edge_df["remove_homo_uniprot1"].tolist()).union(set(proc_edge_df["remove_homo_uniprot2"].tolist()))
    uniprot_ids = [_id.upper() for _id in uniprot_ids]
    print(f"Total number of uniprot ids: {len(uniprot_ids)}")

    df = build_dataset(uniprot_ids)
    df['node_id'] = df.apply(
        lambda row: f"{str(row['uniprot_id']).lower()}_{row['position']}_{row['residuetype']}", 
        axis=1
    )
    df = pd.concat([df.iloc[:,-1], df.iloc[:,:-1]], axis=1)
    # Save to CSV
    df.to_csv("data/proc_data/gnomad_mutation_counts_freq.csv", index=False)