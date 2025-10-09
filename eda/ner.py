import pandas as pd
import ast 
from typing import Set, List
import spacy
import matplotlib.pyplot as plt
from collections import Counter
import os 


NLP_MODEL = "en_core_web_sm"
COMBINDED_DF_CSV = "data/combined.csv"
ENTITY_NER_CSV = "data/combined_with_entity_ner.csv"
IMAGE_PATH = "eda/visualization/ner"
IMAGE_TOP_FREQ_PATH = "eda/visualization/ner/top_freq"

class NERAnalysis:

    DEFAULT_ENTITY_NER_FILE = "data/combined_with_entity_ner.csv"


    def __init__(self, df: pd.DataFrame, description_kws: Set[str], nlp=None):
        self.df = df
        self.nlp = nlp
        self.description_kws = description_kws
        self.ner_df = None
        if not os.path.exists(IMAGE_PATH): os.makedirs(IMAGE_PATH)
        if not os.path.exists(IMAGE_TOP_FREQ_PATH): os.makedirs(IMAGE_TOP_FREQ_PATH)

    def set_ner_df(self, df: pd.DataFrame):
        self.ner_df = df
    
    def set_ner_by_csv(self, file: str):
        self.ner_df = pd.read_csv(file)
    
    def export_ner_to_csv(self, file: str = None):
        if file == None: file = self.DEFAULT_ENTITY_NER_FILE
        self.ner_df.to_csv(file , index= False)

    def _find_cols_keyword(self, keywords: Set[str]) -> List[str]:
        cols = []
        for col in self.df.columns:
            col_parts = col.split('_')
            if any(p.lower() in keywords for p in col_parts) or col.lower() in (kw.lower() for kw in keywords):
                if len(cols) != 0:
                    print(f"Found Another Potential Column: {col}")
                else:
                    print(f"Main Column: {col}")
                cols.append(col)
        return cols

    def _get_descriptions(self, consolidate=False) -> List[str]:
        description_cols = self._find_cols_keyword(self.description_kws)
        if len(description_cols) == 0:
            raise ValueError("Description column not found")

        # Consolidate if multiple description columns exist (not supported rn)
        if len(description_cols) > 1 and consolidate:
            main_col = description_cols[0]
            self.df[main_col] = self.df[description_cols].agg(' '.join, axis=1)
            return [main_col]
        return description_cols

    def _extract_by_label(self, doc, label):
        return [ent.text for ent in doc.ents if ent.label_ == label]

    def extract_entities(self) -> pd.DataFrame:
        desc_cols = self._get_descriptions()
        results = []

        for col in desc_cols:
            print(f"Extracting entities from column: {col}")
            self.df[f"{col}_entities"] = self.df[col].fillna("").apply(
                lambda text: [(ent.text, ent.label_) for ent in self.nlp(text).ents]
            )
            results.append(f"{col}_entities")
        
        return self.df


    def explode_entities(self, entity_cols: List[str] = None) -> pd.DataFrame:
        """
        Flatten all *_entities columns into a single DataFrame with entity_text and entity_label.
        """
        if self.ner_df is None: raise ValueError("NER dataframe is not set. Run extract_entities() or set_ner_by_csv() first.")
        
        if entity_cols is None: entity_cols = [c for c in self.ner_df.columns if c.endswith("_entities")]

        all_entities = []

        for col in entity_cols:
            df_copy = self.ner_df[[col]].copy()

            def parse_entities(x):
                if pd.isna(x):
                    return []
                if isinstance(x, str):
                    try:
                        return ast.literal_eval(x)
                    except:
                        return []
                return x

            df_copy[col] = df_copy[col].apply(parse_entities)
            df_copy = df_copy[df_copy[col].apply(lambda x: isinstance(x, list))]
            df_copy = df_copy.explode(col).dropna()
            df_copy = df_copy[df_copy[col].apply(lambda x: isinstance(x, (list, tuple)) and len(x) == 2)]

            if not df_copy.empty:
                df_copy[["entity_text", "entity_label"]] = pd.DataFrame(
                    df_copy[col].tolist(), index=df_copy.index
                )
                all_entities.append(df_copy[["entity_text", "entity_label"]])

        if not all_entities:
            raise ValueError("No valid entity data found. Check if *_entities columns contain parsed tuples.")

        return pd.concat(all_entities, ignore_index=True)

    def plot_entity_label_distribution(self, entity_cols: List[str] = None, top_n: int = 20):
        """
        Plot the frequency of each entity label.
        """
        plot_name = "entity_frequency"
        entities_long = self.explode_entities(entity_cols)
        label_counts = entities_long['entity_label'].value_counts().head(top_n)

        plt.figure(figsize=(10, 5))
        label_counts.plot(kind='bar', color='skyblue')
        plt.title("Entity Type Frequency")
        plt.ylabel("Count")
        plt.xlabel("Entity Label")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{IMAGE_PATH}/{plot_name}", dpi=300)
        #plt.show()

    def plot_top_entities(self, entity_cols: List[str] = None, top_n: int = 20):
        """
        Plot the most frequent entity texts overall.
        """
        plot_name = 'most_frequency_entities'
        entities_long = self.explode_entities(entity_cols)
        top_entities = entities_long['entity_text'].value_counts().head(top_n)
        plt.figure(figsize=(10, 6))
        top_entities.plot(kind='barh', color='lightgreen')
        plt.title(f"Top {top_n} Most Frequent Entities")
        plt.xlabel("Frequency")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{IMAGE_PATH}/{plot_name}", dpi=300)
        #plt.show()

    def plot_top_entities_per_label(self, entity_cols: List[str] = None, top_n: int = 5):
        """
        Show the most common entities for each label.
        """
        entities_long = self.explode_entities(entity_cols)
        grouped = (
            entities_long.groupby(['entity_label', 'entity_text'])
            .size()
            .reset_index(name='count')
            .sort_values(['entity_label', 'count'], ascending=[True, False])
        )

        for label in grouped['entity_label'].unique():
            top = grouped[grouped['entity_label'] == label].head(top_n)
            plt.figure(figsize=(8, 4))
            plt.barh(top['entity_text'], top['count'], color='orange')
            plt.title(f"Top {top_n} Entities for {label}")
            plt.xlabel("Count")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{IMAGE_TOP_FREQ_PATH}/{label}", dpi=300)
            #plt.show()


# Found Another Potential Column: LIKELIHOOD_DESCRIPTION
# Found Another Potential Column: DESCRIPTION
# Found Another Potential Column: DAMAGE_DESCRIPTION
# Found Another Potential Column: PERSON_RESPONSIBLE_COST_CENTER_DESCRIPTION

if __name__ == "__main__":
    # Check if using GPU
    print("Using GPU:", spacy.prefer_gpu())

    # Set keywords to identify description columns
    description_kws = {"description", "desc"}

    # Load model & data
    nlp = spacy.load(NLP_MODEL)
    df = pd.read_csv(COMBINDED_DF_CSV)

    analyzer = NERAnalysis(df, description_kws, nlp)

    # UNCOMMENT AND RUN THIS FIRST 
    # df_with_entities = analyzer.extract_entities()
    # analyzer.set_ner_df(df_with_entities)
    # analyzer.export_ner_to_csv("data/cer.csv")


    # Change Param to File path where you saved df 
    analyzer.set_ner_by_csv(ENTITY_NER_CSV)

    analyzer.plot_entity_label_distribution(entity_cols=["DESCRIPTION_entities"])
    analyzer.plot_top_entities(entity_cols=["DESCRIPTION_entities"])
    analyzer.plot_top_entities_per_label(entity_cols=["DESCRIPTION_entities"])



    
