## method 1 : entity_conistency_eval.py
use sentence-transformers(all-MiniLM-L6-v2) & k-means method
metrics: Unique Entity Rate & Type Consistency Rate


## method 2 : topology robustness (evaluate_all.py)
metrics: Giant Component Ratio, #components, avg/median component size, avg shortest path, diameter, clustering coefficient, transitivity, #communities, modularity, degree stats, assortativity
Analyze the structural robustness of the graph, including connectivity, clustering, community structure, degree distribution, etc., to reveal whether the graph forms an effective information flow framework, whether it is fragmented, and whether there are functional clusters.


## method 3: semantic_similar_distance.py
metrics: spearman_r & p_value
Text similarity vs graph distance
Verify that "the more semantically similar the nodes are, the closer they are on the graph." Calculate the Spearman correlation between semantic similarity (embedding cosine or surface overlap of node labels) and the shortest path distance (expected to be a significant negative correlation)
Randomly sample node pairs and calculate sim(label_u,label_v) and dist_G(u,v); **The more negative the correlation (the larger the absolute value), the better the "semantic alignment" of the graph.**

## how to run
python evaluate_all.py KG1 (the place where include nodes.csv and edges.csv) 