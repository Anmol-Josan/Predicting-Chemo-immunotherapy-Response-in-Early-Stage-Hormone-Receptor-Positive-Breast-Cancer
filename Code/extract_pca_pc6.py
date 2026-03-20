import sys
from pathlib import Path
adata_path = r"c:\\Users\\ajosan\\Downloads\\Unsupervised-Learning-For-HR-Breast-Cancer-RNA-Sequencing\\Final\\Processed_Data\\processed_s_rna_seq_data.h5ad"
try:
    import anndata as ad
    import numpy as np
except Exception as e:
    print("ERROR_IMPORT", e)
    sys.exit(1)
try:
    adata = ad.read_h5ad(adata_path)
except Exception as e:
    print("ERROR_LOADING", e)
    sys.exit(1)
# Check PCA in adata.uns
if 'pca' in adata.uns:
    pca_uns = adata.uns['pca']
    varr = None
    for k in ['variance_ratio', 'variance_ratio_','variance_ratio_ratio','variance_ratio']:
        if k in pca_uns:
            varr = pca_uns[k]
            break
    print("HAS_PCA")
    if varr is not None:
        try:
            vr = list(map(float, list(varr)))
            print("VARIANCE_RATIO_LEN", len(vr))
            print("VARIANCE_RATIO_FIRST10", vr[:10])
            if len(vr) >= 6:
                print("PC6_VAR", vr[5])
        except Exception as e:
            print("VARIANCE_ERROR", e)
    # try to get PC loadings
    PCs = None
    for k in ['PCs','components','loadings','PCs']:
        if k in pca_uns:
            PCs = pca_uns[k]
            break
    if PCs is not None:
        pcs = np.array(PCs)
        print("PCs_shape", pcs.shape)
        if pcs.shape[0] >= 6:
            pc6 = pcs[5]
            genes = list(adata.var_names) if hasattr(adata, 'var_names') else [str(i) for i in range(len(pc6))]
            idx_pos = np.argsort(-pc6)[:10]
            idx_neg = np.argsort(pc6)[:10]
            print("TOP_POS_GENES", [(genes[int(i)], float(pc6[int(i)])) for i in idx_pos])
            print("TOP_NEG_GENES", [(genes[int(i)], float(pc6[int(i)])) for i in idx_neg])
    else:
        print("NO_PCA_LOADINGS_FOUND")
else:
    print("NO_PCA_IN_UNS")
# Check sample PC scores in obsm
if 'X_pca' in adata.obsm:
    Xp = adata.obsm['X_pca']
    print("X_pca_shape", Xp.shape)
    if Xp.shape[1] >= 6:
        pc6_scores = Xp[:,5]
        print("PC6_scores_summary", float(np.mean(pc6_scores)), float(np.std(pc6_scores)), float(np.min(pc6_scores)), float(np.max(pc6_scores)))
    else:
        print("X_PCA_TOO_FEW_COMPONENTS")
else:
    print("NO_X_PCA_IN_OBSM")
# Numeric obs correlations
num_cols = []
for c in adata.obs.columns:
    try:
        arr = np.array(adata.obs[c].astype(float))
        if not np.isnan(arr).all():
            num_cols.append(c)
    except Exception:
        continue
print("NUMERIC_OBS_COLS", num_cols[:50])
if 'X_pca' in adata.obsm and len(num_cols) > 0:
    import math
    pc6_scores = adata.obsm['X_pca'][:,5]
    for c in num_cols:
        try:
            arr = np.array(adata.obs[c].astype(float))
            if len(arr) == len(pc6_scores):
                # compute pearson
                mask = ~np.isnan(arr)
                if mask.sum() < 3:
                    continue
                r = np.corrcoef(pc6_scores[mask], arr[mask])[0,1]
                print("CORR", c, float(r))
        except Exception:
            continue
