Dictionary Annealing Algorithm in MATLAB for Nearest Neighbor Search

ArXiv Links:
[HCLAE: High Capacity Locally Aggregating Encodings for Approximate Nearest Neighbor Search](http://arxiv.org/pdf/1509.05194)
[Learning Better Encoding for Approximate Nearest Neighbor Search with Dictionary Annealing](http://arxiv.org/pdf/1507.01442)

Usage:
1. Prepare a dataset, e.g, SIFT-1M, in N*dim matrix
2. Learn dictionaries: [~, dict]=DA(learn, [], 8, 256, 10,5)
3. Encode: [idx, epsi, residue]=encode(database,dict,10)