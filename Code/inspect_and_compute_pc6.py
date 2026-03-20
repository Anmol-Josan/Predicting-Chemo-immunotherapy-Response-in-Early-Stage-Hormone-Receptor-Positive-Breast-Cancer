import sys
from pathlib import Path
import numpy as np

try:
    import anndata as ad
    import scipy.sparse as sps
except Exception as e:
    print("IMPORT_ERROR_ANNDATA_SCIPY", e)
    sys.exit(1)

try:
    from sklearn.decomposition import PCA, TruncatedSVD
except Exception as e:
    print("IMPORT_ERROR_SKLEARN", e)
    sys.exit(2)

adata_path = Path(r"c:\Users\ajosan\Downloads\Unsupervised-Learning-For-HR-Breast-Cancer-RNA-Sequencing\Final\Processed_Data\processed_s_rna_seq_data.h5ad")
if not adata_path.exists():
    print("H5AD_NOT_FOUND", adata_path)
    sys.exit(3)

try:
    adata = ad.read_h5ad(str(adata_path))
except Exception as e:
    print("READ_ERROR", e)
    sys.exit(4)

print("PYTHON_EXEC", sys.executable)
print("PYTHON_VERSION", sys.version.split()[0])
print("AD_SHAPE", adata.shape)
print("N_OBS", getattr(adata, 'n_obs', adata.shape[0]), "N_VARS", getattr(adata, 'n_vars', adata.shape[1]))
print("UNS_KEYS", list(adata.uns.keys()))
print("OBSM_KEYS", list(adata.obsm.keys()))
print("VARM_KEYS", list(adata.varm.keys()))
print("LAYERS_KEYS", list(adata.layers.keys()))
print("RAW_PRESENT", hasattr(adata, 'raw') and adata.raw is not None)
try:
    is_sparse = sps.issparse(adata.X)
except Exception:
    is_sparse = False
print("X_IS_SPARSE", is_sparse)
print("FIRST_10_VAR_NAMES", list(map(str, adata.var_names[:10])))

found_pca = False
# check obsm
for k in adata.obsm.keys():
    if 'pca' in k.lower() or 'svd' in k.lower() or ('x_' in k.lower() and 'pca' in k.lower()):
        arr = np.array(adata.obsm[k])
        print("FOUND_OBSM_PCA", k, arr.shape)
        if arr.shape[1] >= 6:
            pc6 = arr[:,5]
            print("PC6_SCORES_SUMMARY", float(np.mean(pc6)), float(np.std(pc6)), float(np.min(pc6)), float(np.max(pc6)))
        found_pca = True

# check uns for pca-like entries
for k, v in adata.uns.items():
    if 'pca' in k.lower() or 'pcs' in k.lower() or 'components' in k.lower():
        print("FOUND_UNS_KEY", k, type(v))
        try:
            if isinstance(v, dict):
                for subk in v:
                    if 'variance' in subk.lower():
                        print("UNS_VARIANCE_KEY", subk, np.array(v[subk]).shape)
            else:
                arr = np.array(v)
                print("UNS_VALUE_SHAPE", arr.shape)
        except Exception as e:
            print("UNS_INSPECT_ERROR", e)

# check varm for stored components
for k in adata.varm.keys():
    try:
        arr = np.array(adata.varm[k])
        print("FOUND_VARM", k, arr.shape)
        # try to interpret shape
        if arr.ndim == 2 and (arr.shape[0] >= 6 or arr.shape[1] >= 6):
            if arr.shape[0] >= 6:
                comp = arr[5]
            else:
                comp = arr[:,5]
            genes = list(map(str, adata.var_names))
            idx_pos = np.argsort(-comp)[:20]
            idx_neg = np.argsort(comp)[:20]
            print("PC6_VARM_TOP_POS", [(genes[int(i)], float(comp[int(i)])) for i in idx_pos])
            print("PC6_VARM_TOP_NEG", [(genes[int(i)], float(comp[int(i)])) for i in idx_neg])
            found_pca = True
    except Exception as e:
        print("VARM_INSPECT_ERROR", k, e)

if not found_pca:
    print("NO_EXISTING_PCA_FOUND")
    # pick matrix
    X = None
    if 'scaled' in adata.layers:
        print("USING_LAYER_scaled")
        X = adata.layers['scaled']
    elif hasattr(adata, 'raw') and adata.raw is not None and getattr(adata.raw, 'X', None) is not None:
        print("USING_RAW")
        X = adata.raw.X
    else:
        print("USING_ADATA_X")
        X = adata.X

    n_samples, n_features = adata.shape
    n_comp = min(20, n_features, n_samples - 1) if n_samples > 1 else min(20, n_features)
    if n_comp < 6:
        print("NOT_ENOUGH_COMPONENTS", n_comp)
        sys.exit(5)

    if sps.issparse(X):
        print("COMPUTING_TruncatedSVD", n_comp)
        svd = TruncatedSVD(n_components=n_comp, random_state=0)
        scores = svd.fit_transform(X)
        comps = svd.components_
    else:
        Xarr = np.asarray(X, dtype=float)
        if np.isnan(Xarr).any():
            col_mean = np.nanmean(Xarr, axis=0)
            inds = np.where(np.isnan(Xarr))
            Xarr[inds] = np.take(col_mean, inds[1])
        Xc = Xarr - np.mean(Xarr, axis=0, keepdims=True)
        print("COMPUTING_PCA", n_comp)
        pca = PCA(n_components=n_comp, random_state=0)
        scores = pca.fit_transform(Xc)
        comps = pca.components_

    pc6_scores = scores[:,5]
    pc6_loadings = comps[5]
    genes = list(map(str, adata.var_names))
    idx_pos = np.argsort(-pc6_loadings)[:20]
    idx_neg = np.argsort(pc6_loadings)[:20]
    print("PC6_COMPUTED_SCORES_SUMMARY", float(np.mean(pc6_scores)), float(np.std(pc6_scores)), float(np.min(pc6_scores)), float(np.max(pc6_scores)))
    print("PC6_TOP_POS_GENES", [(genes[int(i)], float(pc6_loadings[int(i)])) for i in idx_pos])
    print("PC6_TOP_NEG_GENES", [(genes[int(i)], float(pc6_loadings[int(i)])) for i in idx_neg])

    # correlations with numeric obs
    num_cols = []
    for c in adata.obs.columns:
        try:
            arr = np.array(adata.obs[c].astype(float))
            if not np.isnan(arr).all():
                num_cols.append(c)
        except Exception:
            continue
    print("NUMERIC_OBS_COLS", num_cols)
    corrs = []
    for c in num_cols:
        arr = np.array(adata.obs[c].astype(float))
        mask = ~np.isnan(arr)
        if mask.sum() < 3:
            continue
        try:
            r = np.corrcoef(pc6_scores[mask], arr[mask])[0,1]
            corrs.append((c, float(r)))
        except Exception:
            continue
    corrs_sorted = sorted(corrs, key=lambda x: -abs(x[1]))[:10]
    print("TOP_OBS_CORRELATIONS", corrs_sorted)

print("DONE")
