"""
Extract triplets for each incident using GraphRAG parquet outputs.

Usage:
    python graphRAG/extract.py 
"""
import pandas as pd
from pathlib import Path


class Extractor:
    INCIDENT_ID_COL = "RECORD_NO_LOSS_POTENTIAL"
    OUTPUT_CSV = "incident_triples.csv"

    # Id's
    ID = "id"
    REL_ID = "relationship_id"
    TEXT_UNIT_ID = "text_unit_id"
    INCIDENT_ID = "incident_id"


    TEXT_UNIT_IDS = "text_unit_ids"


    def _rename_row(self, df: pd.DataFrame, col : str, rename_option: str):
        if col not in df:
            return 
        df = df.rename(columns = {col: rename_option})
        return df
    
    def __init__(self, output_dir:Path, incident_csv: Path | None = None):
        self.output_dir = output_dir
        self.incident_csv = incident_csv or (output_dir.parent / "input" / "dev_sample.csv")

        # Parquet Files 
        self.entities = pd.read_parquet(output_dir / "entities.parquet")
        self.relationships = pd.read_parquet(output_dir / "relationships.parquet")
        self.text_units = pd.read_parquet(output_dir / "text_units.parquet")
        self.documents = pd.read_parquet(output_dir / "documents.parquet")

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
    
    def normalize_df():
        pass

    def view_df(self, df: pd.DataFrame, name: str):
        print(f"Viewing DataFrame : {name}")
        print(df.head(10).to_string())
        print(df.columns)
        print('\n')

        
    def merge(self):
        """
        Gather Triplets from Entities & Relations -> Match to Text Units -> Match to Documents

        It seems that we don't need to merge to documents? Can't we just merge along Text Units as it contains the 
        RECORD_NO_LOSS_POTENTIAL which corresponds to a singular incident?
        """

        # Rename ID For Relationships & Explode Text Unit IDs
        rel_df = self._rename_row(self.relationships, self.ID, self.REL_ID)
        rel_df = rel_df.explode(self.TEXT_UNIT_IDS).dropna(subset = [self.TEXT_UNIT_IDS])
        rel_df = self._rename_row(rel_df, self.TEXT_UNIT_IDS, self.TEXT_UNIT_ID)

        # Rename ID For Documents & Explode Document IDs
        tu_df = self._rename_row(self.text_units, self.ID, self.TEXT_UNIT_ID)
        tu_df = self._rename_row(tu_df, self.INCIDENT_ID_COL, self.INCIDENT_ID)

        # Merge
        merged = rel_df.merge(tu_df, on = self.TEXT_UNIT_ID, how="left").dropna(subset=[self.INCIDENT_ID])
        colsToKeep = [self.INCIDENT_ID, self.source_col, self.rel_col, self.target_col, self.TEXT_UNIT_ID, self.REL_ID, "document_ids"]
        merged = merged[colsToKeep].sort_values([self.INCIDENT_ID])


        # Save 
        csv_path = self.output_dir / self.OUTPUT_CSV
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(csv_path, index=False)
        print(f"Saved flattened triples to {csv_path} (rows={len(merged)})")
        return merged


def main():
    out_dir = Path("graphRAG/output")
    extractor = Extractor(out_dir)
    extractor.merge()


if __name__ == "__main__":
    main()