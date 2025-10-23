import pandas as pd
import spacy
import textacy.extract as xt
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

def extract_svo(text):
    doc = nlp(text)
    out = []
    for s, v, o in xt.subject_verb_object_triples(doc):
        out.append((
            " ".join([t.text for t in s]),
            " ".join([t.text for t in v]),
            " ".join([t.text for t in o])
        ))
    return out

df = pd.read_csv("/projects/dsci435/fmcsafetyevents_fa25/data/cleaned_description_translated.csv")

rows = []
for r in tqdm(df.itertuples(), total=len(df), desc="Extracting SVO triples"):
    for s, v, o in extract_svo(r.text):
        rows.append({
            "record_no": r.RECORD_NO_LOSS_POTENTIAL,
            "subject": s,
            "relation": v,
            "object": o
        })

kg_df = pd.DataFrame(rows)
print(kg_df.head())

kg_df.to_csv("knowledge_graph_edges.csv", index=False, encoding="utf-8")
print("Saved knowledge_graph_edges.csv")