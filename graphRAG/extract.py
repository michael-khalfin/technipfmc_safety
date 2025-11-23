"""
Extract triplets for each incident using GraphRAG parquet outputs.

Usage:
    python graphRAG/extract.py 
"""
import pandas as pd
from pathlib import Path

DESCRIPTIVE  = False
ENTITIES_FILE = "entities_filtered.parquet"
RELATIONS_FILE = "relationships_filtered.parquet"

class Extractor:
    INCIDENT_ID_COL = "RECORD_NO_LOSS_POTENTIAL"
    OUTPUT_CSV = "incident_triples.csv"

    ID = "id"
    REL_ID = "relationship_id"
    TEXT_UNIT_ID = "text_unit_id"
    TEXT_UNIT_IDS = "text_unit_ids"
    DOC_ID = "document_id"
    DOC_IDS = "document_ids"
    INCIDENT_ID = "incident_id"
    


    def _rename_row(self, df: pd.DataFrame, col : str, rename_option: str):
        if col not in df:
            return 
        df = df.rename(columns = {col: rename_option})
        return df
    
    def __init__(self, output_dir:Path, isDescriptive: bool ):
        self.output_dir = output_dir
        self.incident_csv = output_dir.parent / "input" / "dev_sample.csv"
        self.isDescriptive = isDescriptive
        self.merged = None

        # Parquet Files 
        self.relationships = pd.read_parquet(output_dir / RELATIONS_FILE)
        self.text_units = pd.read_parquet(output_dir / "text_units.parquet")
        self.documents = pd.read_parquet(output_dir / "documents.parquet")
        self.entities = pd.read_parquet(output_dir / ENTITIES_FILE)

        # Columns For Relationships 
        self.source_col = "source"
        self.target_col = "target"
        self.rel_col = "description"
        if not self.source_col or not self.target_col or not self.rel_col:
            raise ValueError("Could not locate source/target columns in relationships parquet.")
    

        # DEBUG: View Relationships
        # self.view_df(self.entities, "Entities")
        # self.view_df(self.relationships, "Relationships")
        # self.view_df(self.text_units, "Text Units")
        # self.view_df(self.documents, "Documents")
    

    def view_df(self, df: pd.DataFrame, name: str):
        print(f"Viewing DataFrame : {name}")
        # print(df.head(10).to_string())
        print(df.head(10))
        print(df.columns)
        print('\n')

    def merge(self):
        """
        Correct mapping for GraphRAG 0.6+
        relationships → text_units → documents → CSV (incident ID)
        """

        # Indexes
        csv_df = pd.read_csv(self.incident_csv)
        csv_df["document_index"] = csv_df.index
        doc_df = self.documents.copy()
        doc_df["document_index"] = doc_df["human_readable_id"] 

        # Join CSV incident IDs into documents
        doc_df = doc_df.merge(
            csv_df[["document_index", self.INCIDENT_ID_COL]],
            on="document_index",
            how="left"
        ).rename(columns={self.INCIDENT_ID_COL: self.INCIDENT_ID})

      
        # Explode Text Unit Id's IN Relationships Df
        rel_df = self.relationships.rename(columns={self.ID: self.REL_ID})
        rel_df = rel_df.explode(self.TEXT_UNIT_IDS).dropna(subset=[self.TEXT_UNIT_IDS])
        rel_df = rel_df.rename(columns={self.TEXT_UNIT_IDS: self.TEXT_UNIT_ID})

        # Explode Document Id's In Text Units Df
        tu_df = self.text_units.rename(columns={self.ID: self.TEXT_UNIT_ID})
        tu_df = tu_df.explode(self.DOC_IDS).rename(columns={self.DOC_IDS: self.DOC_ID})
        
        # Merge relationships -> text_units
        merged = rel_df.merge(tu_df, on=self.TEXT_UNIT_ID, how="left")
        
        # Add incident_id from documents
        doc_df_small = doc_df.rename(columns={self.ID: self.DOC_ID})
        merged = merged.merge(
            doc_df_small[[self.DOC_ID, self.INCIDENT_ID]],
            on=self.DOC_ID,
            how="left"
        )

        # Clean Merged 
        merged = merged.dropna(subset=[self.INCIDENT_ID])
        cols = [
            self.INCIDENT_ID,
            self.source_col,
            self.rel_col,
            self.target_col,
            self.TEXT_UNIT_ID,
            self.REL_ID,
            self.DOC_ID
        ]
        merged = merged[cols].sort_values([self.INCIDENT_ID])

        # Save
        csv_path = self.output_dir / self.OUTPUT_CSV
        merged.to_csv(csv_path, index=False)
        print(f"Saved flattened triples to {csv_path} (rows={len(merged)})\n")
        self.merged = merged
        return merged


    # def merge(self):
    #     """
    #     Gather Triplets from Entities & Relations -> Match to Text Units -> Match to Documents

    #     It seems that we don't need to merge to documents? Can't we just merge along Text Units as it contains the 
    #     RECORD_NO_LOSS_POTENTIAL which corresponds to a singular incident?
    #     """

    #     # Rename ID For Relationships & Explode Text Unit IDs
    #     rel_df = self._rename_row(self.relationships, self.ID, self.REL_ID)
    #     rel_df = rel_df.explode(self.TEXT_UNIT_IDS).dropna(subset = [self.TEXT_UNIT_IDS])
    #     rel_df = self._rename_row(rel_df, self.TEXT_UNIT_IDS, self.TEXT_UNIT_ID)

    #     # Rename ID For Documents & Explode Document IDs
    #     tu_df = self._rename_row(self.text_units, self.ID, self.TEXT_UNIT_ID)
    #     tu_df = self._rename_row(tu_df, self.INCIDENT_ID_COL, self.INCIDENT_ID)

    #     # Merge
    #     merged = rel_df.merge(tu_df, on = self.TEXT_UNIT_ID, how="left").dropna(subset=[self.INCIDENT_ID])
    #     colsToKeep = [self.INCIDENT_ID, self.source_col, self.rel_col, self.target_col, self.TEXT_UNIT_ID, self.REL_ID, "document_ids"]
    #     merged = merged[colsToKeep].sort_values([self.INCIDENT_ID])


    #     # Save 
    #     csv_path = self.output_dir / self.OUTPUT_CSV
    #     csv_path.parent.mkdir(parents=True, exist_ok=True)
    #     merged.to_csv(csv_path, index=False)
    #     print(f"Saved flattened triples to {csv_path} (rows={len(merged)})")
    #     self.merged = merged
    #     return merged

    def describe(self, inDepth = False):
        if self.merged is None:
            print("Must run merge() before describe().")
            return 
        
        df = self.merged
        df["triplet"] = list(zip(df[self.source_col], df[self.rel_col], df[self.target_col]))
        grouped = df.groupby(self.INCIDENT_ID)["triplet"].apply(list)
        totalTriplets = 0

        if inDepth: print("\n=== Triplets per Incident ===")
        for incident_id, triplets in grouped.items():
            totalTriplets += len(triplets)
            if (inDepth):
                print(f"\nIncident: {incident_id}  (Triplets: {len(triplets)})")
                for s, r, t in triplets:
                    print(f"  - ({s}) -[{r}]-> ({t})")
            
        print(f"Average Triplets Per Incident: {totalTriplets // len(grouped)}")
        return grouped.to_dict()

def main():
    out_dir = Path("graphRAG/output")
    extractor = Extractor(out_dir, DESCRIPTIVE)
    extractor.merge()
    extractor.describe(inDepth= False)


if __name__ == "__main__":
    main()