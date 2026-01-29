import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA


class AxisNetMicrobiomeLoader:
    def __init__(self, base_loader):
        self.base = base_loader
        self.microbiome_data = None
        self.microbiome_features = None
        self.paired_indices = None

    def load_multimodal(self, connectivity="correlation", atlas="ho", microbiome_path=None,
                        similarity_threshold=0.8, top_k=5, microbiome_pca_dim=64,
                        drop_age=False, drop_sex=False):
        print("  Loading fMRI data...")
        fmri_data, labels, clinical_data = self.base.load_data(
            connectivity, atlas, drop_age=drop_age, drop_sex=drop_sex
        )
        n_fmri = len(fmri_data)
        print(f"  fMRI sample count: {n_fmri}")

        if microbiome_path is not None:
            print("  Loading microbiome data...")
            microbiome_meta, microbiome_raw = self._load_microbiome_data(microbiome_path)
            print(f"  Microbiome sample count: {len(microbiome_meta) if microbiome_meta is not None else 0}")
        else:
            microbiome_meta, microbiome_raw = None, None

        if microbiome_meta is not None:
            print("  Creating enhanced microbiome features...")
            microbiome_features = self._create_microbiome_features(
                fmri_data, labels, clinical_data, microbiome_meta, microbiome_raw,
                similarity_threshold, top_k, microbiome_pca_dim,
                drop_age=drop_age, drop_sex=drop_sex
            )
        else:
            print("  Generating simulated microbiome features...")
            microbiome_features = self._create_simulated_microbiome_features(n_fmri, microbiome_pca_dim)

        return fmri_data, labels, clinical_data, microbiome_features

    def _load_microbiome_data(self, microbiome_path):
        try:
            path = Path(microbiome_path)
            meta_path = self._find_meta_path(path)
            microbiome_meta = pd.read_csv(meta_path) if meta_path else None

            microbiome_raw = None
            sample_ids = None
            if path.suffix.lower() in {".biom"}:
                microbiome_raw, sample_ids = self._load_biom_features(path)
            elif path.suffix.lower() in {".hdf5", ".h5"}:
                microbiome_raw, sample_ids = self._load_hdf5_features(path)
                if microbiome_raw is None:
                    alt_biom = path.parent / "feature-table.biom"
                    if alt_biom.exists():
                        microbiome_raw, sample_ids = self._load_biom_features(alt_biom)
                if microbiome_raw is None:
                    alt_h5 = path.parent / "microbe.hdf5"
                    if alt_h5.exists():
                        microbiome_raw, sample_ids = self._load_hdf5_features(alt_h5)
            else:
                df = pd.read_csv(path)
                meta_cols = [
                    "ID", "AGE_AT_SCAN", "SEX", "DX_GROUP", "Control_Type",
                    "Cohort", "Subjects_Location", "Variable_Region", "Match_IDs",
                ]
                available_meta = [c for c in meta_cols if c in df.columns]
                microbiome_meta = df[available_meta].copy() if available_meta else microbiome_meta
                feature_cols = [c for c in df.columns if c not in meta_cols]
                if feature_cols:
                    microbiome_raw = df[feature_cols].values.astype(np.float32)
                    sample_ids = df["ID"].astype(str).tolist() if "ID" in df.columns else None

            microbiome_meta, microbiome_raw = self._align_meta_and_features(
                microbiome_meta, microbiome_raw, sample_ids
            )
            return microbiome_meta, microbiome_raw
        except Exception as e:
            print(f"Loading microbiome data failed: {e}")
            return None, None

    def _find_meta_path(self, microbiome_path):
        candidates = [
            microbiome_path.parent / "microbe_data.csv",
            Path(__file__).resolve().parent / "microbe_data.csv",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _load_biom_features(self, biom_path):
        try:
            from biom import load_table
            table = load_table(str(biom_path))
            sample_ids = [str(s) for s in table.ids(axis="sample")]
            features = table.matrix_data.T.toarray().astype(np.float32)
            return features, sample_ids
        except Exception as e:
            print(f"BIOM loading failed: {e}")
            return None, None

    def _load_hdf5_features(self, h5_path):
        try:
            import h5py
            with h5py.File(h5_path, "r") as f:
                keys = list(f.keys())
                if not keys:
                    return None, None
                sample_ids = [str(k) for k in keys]
                features = np.stack([f[k][:] for k in keys]).astype(np.float32)
            return features, sample_ids
        except Exception as e:
            print(f"HDF5加载失败: {e}")
            return None, None

    def _align_meta_and_features(self, microbiome_meta, microbiome_raw, sample_ids):
        if microbiome_raw is None or sample_ids is None:
            return microbiome_meta, microbiome_raw
        if microbiome_meta is None or "ID" not in microbiome_meta.columns:
            microbiome_meta = pd.DataFrame({"ID": sample_ids})
            return microbiome_meta, microbiome_raw

        meta_indexed = microbiome_meta.set_index("ID")
        aligned_meta = meta_indexed.reindex(sample_ids).reset_index()
        return aligned_meta, microbiome_raw

    def _create_microbiome_features(self, fmri_data, labels, clinical_data,
                                    microbiome_meta, microbiome_raw,
                                    similarity_threshold, top_k, microbiome_pca_dim,
                                    drop_age=False, drop_sex=False):
        n_fmri = len(fmri_data)
        n_microbiome = len(microbiome_meta) if microbiome_meta is not None else 0

        micro_features = self._prepare_microbiome_features(microbiome_raw, microbiome_pca_dim, n_microbiome)
        if micro_features is None:
            print("  No real microbiome features were provided, and simulated features were used for graph regularization.")
            micro_features = self._generate_mock_microbiome_data(n_microbiome, n_features=microbiome_pca_dim)

        enhanced_features = np.zeros((n_fmri, micro_features.shape[1]), dtype=np.float32)

        if microbiome_meta is not None and n_microbiome > 0:
            micro_age = microbiome_meta.get("AGE_AT_SCAN", pd.Series([0] * n_microbiome))
            micro_sex = microbiome_meta.get("SEX", pd.Series([0] * n_microbiome))
            micro_age = micro_age.fillna(0).astype(float).values
            micro_sex = micro_sex.fillna(0).astype(int).values

            fmri_age = clinical_data[:, 2].astype(float)
            fmri_sex = clinical_data[:, 1].astype(int)

            for i in range(n_fmri):
                if drop_sex:
                    candidate_idx = np.arange(n_microbiome)
                else:
                    same_sex = np.where(micro_sex == fmri_sex[i])[0]
                    candidate_idx = same_sex if len(same_sex) > 0 else np.arange(n_microbiome)

                if drop_age:
                    rng = np.random.RandomState(42 + i)
                    top_indices = rng.choice(candidate_idx, size=min(top_k, len(candidate_idx)), replace=False)
                else:
                    age_diff = np.abs(micro_age[candidate_idx] - fmri_age[i])
                    sorted_idx = candidate_idx[np.argsort(age_diff)]
                    top_indices = sorted_idx[:top_k]

                selected_micro = micro_features[top_indices]
                enhanced_features[i] = selected_micro.mean(axis=0)

        return enhanced_features.astype(np.float32)

    def _prepare_microbiome_features(self, microbiome_raw, microbiome_pca_dim, n_microbiome):
        if microbiome_raw is None or n_microbiome == 0:
            return None

        micro = microbiome_raw.copy().astype(np.float32)
        micro = np.maximum(micro, 1e-6)
        log_micro = np.log(micro)
        log_mean = log_micro.mean(axis=1, keepdims=True)
        clr = log_micro - log_mean

        n_components = min(microbiome_pca_dim, clr.shape[1], max(1, clr.shape[0] - 1))
        if n_components <= 0:
            return None

        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(clr).astype(np.float32)

    def _create_simulated_microbiome_features(self, n_fmri, feature_dim):
        print(f"  Generating simulated microbiome features: {n_fmri} samples × {feature_dim} features")
        features = np.random.randn(n_fmri, feature_dim).astype(np.float32)
        for i in range(n_fmri):
            base_pattern = np.sin(np.linspace(0, 4 * np.pi, feature_dim)) * 0.5
            noise = np.random.randn(feature_dim) * 0.3
            features[i] = base_pattern + noise
        return features

    def _generate_mock_microbiome_data(self, n_samples, n_features=2503):
        data = np.random.exponential(0.1, (n_samples, n_features))
        data = data / data.sum(axis=1, keepdims=True)
        return data.astype(np.float32)
