import pandas as pd
import spacy
from tqdm import tqdm

# --- Configuration ---
INPUT_CSV_PATH = 'data/cleaned_data.csv'
OUTPUT_CSV_PATH = 'manual_knowledge_graph/kg_relations.csv'
TEXT_COLUMN_TO_PROCESS = 'TITLE' 
UNIQUE_ID_COLUMN = 'INCIDENT_NO'
SPACY_MODEL = 'en_core_web_md'

# --- Development Mode ---
# Set to True to run on a small sample for fast testing
DEV_MODE = True
DEV_ROW_LIMIT = 1000

# --- KG Schema Definition ---
# This is a starting point based on the sample text. We can expand this.
entity_patterns = [
    # Equipment
    {"label": "EQUIPMENT", "pattern": "forklift", "id": "Equipment"},
    {"label": "EQUIPMENT", "pattern": "crane", "id": "Equipment"},
    {"label": "EQUIPMENT", "pattern": "hammer", "id": "Equipment"},
    {"label": "EQUIPMENT", "pattern": "tripod", "id": "Equipment"},
    {"label": "EQUIPMENT", "pattern": "slings", "id": "Equipment"},
    {"label": "EQUIPMENT", "pattern": "valve", "id": "Equipment"},
    {"label": "EQUIPMENT", "pattern": "spool", "id": "Equipment"},
    {"label": "EQUIPMENT", "pattern": [{"LOWER": "pipe"}, {"LOWER": "spool"}], "id": "Equipment"},
    {"label": "EQUIPMENT", "pattern": "compressor", "id": "Equipment"},
    {"label": "EQUIPMENT", "pattern": [{"LOWER": "hot"}, {"LOWER": "stab"}, {"LOWER": "dummy"}], "id": "Equipment"},
    
    # Body Parts / Injuries
    {"label": "BODY_PART", "pattern": "finger", "id": "BodyPart"},
    {"label": "BODY_PART", "pattern": "leg", "id": "BodyPart"},
    {"label": "BODY_PART", "pattern": "shins", "id": "BodyPart"},
    {"label": "INJURY", "pattern": "cut", "id": "Injury"},
    {"label": "INJURY", "pattern": "fracture", "id": "Injury"},
    {"label": "INJURY", "pattern": "scalded", "id": "Injury"},

    # People / Roles
    {"label": "PERSON", "pattern": "contractor", "id": "Role"},
    {"label": "PERSON", "pattern": "employee", "id": "Role"},
    {"label": "PERSON", "pattern": "engineer", "id": "Role"},
    {"label": "PERSON", "pattern": "crew member", "id": "Role"},
]

def extract_relations(doc):
    """
    Extracts relations between named entities.
    A relation is a tuple: (subject_text, subject_label, relation, object_text, object_label)
    """
    relations = []
    for ent1 in doc.ents:
        for ent2 in doc.ents:
            if ent1 == ent2:
                continue
            # Find a common ancestor verb between the two entities
            ancestor = next((tok for tok in ent1.root.ancestors if tok in ent2.root.ancestors), None)
            if ancestor and ancestor.pos_ == "VERB":
                # Check for a subject/object relationship
                if "subj" in ent1.root.dep_ and "obj" in ent2.root.dep_:
                     relations.append({
                         "subject": ent1.text,
                         "subject_type": ent1.label_,
                         "relation": ancestor.lemma_,
                         "object": ent2.text,
                         "object_type": ent2.label_
                     })
    return relations

def main():
    print("--- Starting Targeted Entity and Relation Extraction ---")

    print(f"Loading spaCy model: {SPACY_MODEL}...")
    nlp = spacy.load(SPACY_MODEL)

    print("Configuring EntityRuler with custom patterns...")
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(entity_patterns)

    print(f"Loading data from '{INPUT_CSV_PATH}'...")
    try:
        read_options = {}
        if DEV_MODE:
            read_options['nrows'] = DEV_ROW_LIMIT
            print(f"** DEVELOPMENT MODE ON: Reading only first {DEV_ROW_LIMIT} rows. **")
        
        # Ensure low_memory=False for mixed-type columns
        df = pd.read_csv(INPUT_CSV_PATH, low_memory=False, **read_options)
        df = df.dropna(subset=[TEXT_COLUMN_TO_PROCESS])

    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_CSV_PATH}'.")
        return
    except KeyError:
        print(f"Error: A required column was not found. Please check TEXT_COLUMN_TO_PROCESS and UNIQUE_ID_COLUMN.")
        return
        
    print(f"Data loaded. Processing {len(df)} rows.")

    print("Extracting relations...")
    tqdm.pandas(desc="Processing text")
    df['relations'] = df[TEXT_COLUMN_TO_PROCESS].progress_apply(lambda text: extract_relations(nlp(str(text))))
    
    print("Formatting and saving results...")
    output_df = df.explode('relations').dropna(subset=['relations'])

    if not output_df.empty:
        relations_df = pd.json_normalize(output_df['relations'])
        
        final_df = pd.concat([
            output_df[[UNIQUE_ID_COLUMN, TEXT_COLUMN_TO_PROCESS]].reset_index(drop=True), 
            relations_df
        ], axis=1)
        final_df = final_df.rename(columns={TEXT_COLUMN_TO_PROCESS: 'original_text'})
    else:
        print("No relations extracted.")
        final_df = pd.DataFrame()

    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"--- Process Finished. Output saved to {OUTPUT_CSV_PATH} ---")
    print(f"Found a total of {len(final_df)} relations.")

if __name__ == '__main__':
    main()