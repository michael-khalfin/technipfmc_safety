"""
Extract triplets for each incident using GraphRAG parquet outputs.

Usage:
    python graphRAG/extract.py 
"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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
        self.viz_dir = self.output_dir / "_viz"
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

    def analyze_triplets(self, top_k: int = 10):
        df = self.merged.copy()

        # TODO: Make this so this only called once throughout the class  
        triplets_per_incident = (
            df.groupby(self.INCIDENT_ID)
            .size()
            .rename("triplet_count")
        )

        # Find da metrics
        mean_count, var_count, std_count = triplets_per_incident.mean(), triplets_per_incident.var(), triplets_per_incident.std()

        print(f"\nTRIPLETS STATISTICS ({self.output_dir})")
        print(f"Incidents: {len(triplets_per_incident)}")
        print(f"Mean: {mean_count:.2f}")
        print(f"Variance: {var_count:.2f}")
        print(f"Std Dev: {std_count:.2f}")
        print(f"Min: {triplets_per_incident.min()}")
        print(f"Max: {triplets_per_incident.max()}")

    
        # Histogram for Triplets Per Incident
        plt.figure(figsize=(10, 6))
        plt.hist(triplets_per_incident, bins=30)
        plt.title("Distribution of Triplets per Incident")
        plt.xlabel("Number of Triplets")
        plt.ylabel("Number of Incidents")
        plt.tight_layout()
        plt.savefig(self.viz_dir / "triplets_per_incident_hist.png")
        plt.close()


        # Top K Predicates (TODO: Make this into a small topk function to reduce repeated code)
        most_common_predicates = df[self.rel_col].value_counts().head( top_k)
        top_predicates = most_common_predicates.sort_values(ascending=True)
        plt.figure(figsize=(10, 6))
        plt.barh(top_predicates.index, top_predicates.values)
        plt.title(f"Top {top_k} Most Frequent Predicates")
        plt.xlabel("Frequency")
        plt.ylabel("Predicate")
        plt.tight_layout()
        plt.savefig(self.viz_dir / "top_predicates.png")
        plt.close()

        # Top K Objects
        most_common_objects = df[self.target_col].value_counts().head(top_k )
        top_objects = most_common_objects.sort_values(ascending=True)
        plt.figure(figsize=(10, 6))
        plt.barh(top_objects.index, top_objects.values)
        plt.title(f"Top {top_k} Most Frequent Objects")
        plt.xlabel("Frequency")
        plt.ylabel("Object")
        plt.tight_layout()
        plt.savefig(self.viz_dir / "top_objects.png")
        plt.close()


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
    extractor.analyze_triplets(top_k=10)


if __name__ == "__main__":
    main()