import pandas as pd
import spacy
from tqdm import tqdm

# --- Configuration ---
INPUT_CSV_PATH = 'data/cleaned_data.csv'
OUTPUT_CSV_PATH = 'manual_knowledge_graph/extracted_triples.csv'
# This is the column we'll process
TEXT_COLUMN_TO_PROCESS = 'TITLE'
# We'll use a medium-sized spaCy model
SPACY_MODEL = 'en_core_web_md'

# --- Main Extraction Logic ---

def extract_triples(text, nlp_model):
    """
    Extracts Subject-Verb-Object triples from a given text using spaCy's dependency parser.

    Args:
        text (str): The text to process.
        nlp_model: The loaded spaCy model.

    Returns:
        list: A list of tuples, where each tuple is a (subject, predicate, object) triple.
    """
    # spaCy can have issues with non-string types, so we ensure the input is a string.
    if not isinstance(text, str) or text.strip() == '':
        return []

    doc = nlp_model(text)
    triples = []

    for token in doc:
        # We are looking for verbs to form the predicate of our triple.
        if "VERB" in token.pos_:
            subject = None
            obj = None

            # Traverse the token's children in the dependency tree to find subject and object.
            for child in token.children:
                # Direct objects and nominal subjects are primary targets.
                if "dobj" in child.dep_ and child.pos_ != 'PUNCT':
                    obj = child.text
                if "nsubj" in child.dep_ and child.pos_ != 'PUNCT':
                    subject = child.text

            # If we found a subject and an object, we have a triple.
            if subject and obj:
                triples.append((subject, token.lemma_, obj)) # Using lemma for the verb

    return triples

def main():
    """
    Main function to run the data loading, processing, and saving.
    """
    print("--- Starting Triple Extraction Process ---")

    # Load the spaCy model
    print(f"Loading spaCy model: {SPACY_MODEL}...")
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        print(f"spaCy model '{SPACY_MODEL}' not found. Downloading...")
        spacy.cli.download(SPACY_MODEL)
        nlp = spacy.load(SPACY_MODEL)
    print("Model loaded successfully.")

    # Load the consolidated data
    print(f"Loading data from '{INPUT_CSV_PATH}'...")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_CSV_PATH}'. Please check the path.")
        return
    print(f"Data loaded. Found {len(df)} rows.")

    # Process the text column
    print(f"Extracting triples from the '{TEXT_COLUMN_TO_PROCESS}' column...")
    
    tqdm.pandas(desc="Processing rows")    
    df['extracted_triples'] = df[TEXT_COLUMN_TO_PROCESS].progress_apply(lambda x: extract_triples(x, nlp))
    
    print("Triple extraction complete.")

    # Format the output
    # We'll "explode" the list of triples into separate rows for a clean CSV output.
    output_df = df.explode('extracted_triples').dropna(subset=['extracted_triples'])
    
    if output_df.empty:
        print("No triples were extracted. The output file will be empty.")
        final_df = pd.DataFrame(columns=['subject', 'predicate', 'object', 'original_text'])
    else:
        # Create separate columns for subject, predicate, and object
        output_df[['subject', 'predicate', 'object']] = pd.DataFrame(
            output_df['extracted_triples'].tolist(), index=output_df.index
        )
        # Select and rename columns for the final output file
        final_df = output_df[['subject', 'predicate', 'object', TEXT_COLUMN_TO_PROCESS]].rename(
            columns={TEXT_COLUMN_TO_PROCESS: 'original_text'}
        )

    # Save the results
    print(f"Saving extracted triples to '{OUTPUT_CSV_PATH}'...")
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"--- Process Finished. Output saved to {OUTPUT_CSV_PATH} ---")
    print(f"Found a total of {len(final_df)} triples.")


if __name__ == '__main__':
    main()
