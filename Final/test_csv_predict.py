import pandas as pd
import numpy as np
from app import load_model, build_feature_vector, predict

model, mean, std, cfg = load_model()

files = ['sample_responder.csv', 'sample_non_responder.csv']

for fn in files:
    print('\n==', fn)
    df = pd.read_csv(fn)
    for idx, row in df.iterrows():
        gene_pca = [row.get(f'gene_pca_{i+1}', 0.0) for i in range(15)]
        cdr3_tra = str(row.get('cdr3_TRA', ''))
        cdr3_trb = str(row.get('cdr3_TRB', ''))
        n_genes = float(row.get('n_genes_by_counts', 0))
        total_cts = float(row.get('total_counts', 0))
        pct_mt_val = float(row.get('pct_counts_mt', 0))

        features = build_feature_vector(gene_pca, cdr3_tra, cdr3_trb,
                                        n_genes, total_cts, pct_mt_val)
        if mean is not None and std is not None:
            features = (features - np.array(mean, dtype=np.float32)) / (np.array(std, dtype=np.float32) + 1e-8)

        pred, conf = predict(model, features)
        print(idx, '->', pred, f'(confidence={conf:.3f})')
