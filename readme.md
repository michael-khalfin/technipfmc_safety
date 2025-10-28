## method 1 : entity_conistency_eval.py
use sentence-transformers(all-MiniLM-L6-v2) & k-means method
metrics: Unique Entity Rate & Type Consistency Rate

## method 2: link_prediction_holdout.py
metrics: PR-AUC/ROC-AUC/P@K/R@K/MAP
Randomly occlude p% of the real edges as the "positive sample - test set"; Then negatively sample the same amount of "negative samples" from the non-edges. 
Score the "candidate edges" using a set of scorers (Adamic-Adar, Jaccard, RA, personalized PageRank, or embedded dot product).

## method 3: semantic_similar_distance.py
metrics: spearman_r & p_value
Text similarity vs graph distance
Verify that "the more semantically similar the nodes are, the closer they are on the graph." Calculate the Spearman correlation between semantic similarity (embedding cosine or surface overlap of node labels) and the shortest path distance (expected to be a significant negative correlation)
Randomly sample node pairs and calculate sim(label_u,label_v) and dist_G(u,v); **The more negative the correlation (the larger the absolute value), the better the "semantic alignment" of the graph.**

## how to run
python evaluate_all.py KG1 (the place where include nodes.csv and edges.csv) 