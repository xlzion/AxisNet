import numpy as np
from sklearn.model_selection import StratifiedKFold

from . import abide_parser as Reader
from ..utils.gcn_utils import preprocess_features


class AxisNetDataLoader:
    def __init__(self, data_folder=None, phenotype_path=None, subject_ids_path=None):
        self.pd_dict = {}
        self.node_ftr_dim = 2000
        self.num_classes = 2
        self.data_folder = data_folder
        self.phenotype_path = phenotype_path
        self.subject_ids_path = subject_ids_path

    def load_data(self, connectivity="correlation", atlas="ho", drop_age=False, drop_sex=False):
        Reader.set_data_paths(self.data_folder, self.phenotype_path, self.subject_ids_path)
        subject_IDs = Reader.get_ids()
        labels = Reader.get_subject_score(subject_IDs, score="DX_GROUP")
        num_nodes = len(subject_IDs)

        sites = Reader.get_subject_score(subject_IDs, score="SITE_ID")
        unique = np.unique(list(sites.values())).tolist()
        ages = Reader.get_subject_score(subject_IDs, score="AGE_AT_SCAN")
        genders = Reader.get_subject_score(subject_IDs, score="SEX")

        y = np.zeros([num_nodes])
        site = np.zeros([num_nodes], dtype=int)
        age = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=int)
        for i in range(num_nodes):
            y[i] = int(labels[subject_IDs[i]])
            site[i] = unique.index(sites[subject_IDs[i]])
            age[i] = float(ages[subject_IDs[i]])
            gender[i] = genders[subject_IDs[i]]

        self.y = y - 1
        self.raw_features = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas)

        phonetic_data = np.zeros([num_nodes, 3], dtype=np.float32)
        phonetic_data[:, 0] = site
        phonetic_data[:, 1] = 0 if drop_sex else gender
        phonetic_data[:, 2] = 0 if drop_age else age

        self.pd_dict["SITE_ID"] = np.copy(phonetic_data[:, 0])
        self.pd_dict["SEX"] = np.copy(phonetic_data[:, 1])
        self.pd_dict["AGE_AT_SCAN"] = np.copy(phonetic_data[:, 2])

        return self.raw_features, self.y, phonetic_data

    def load_multimodal_data(self, connectivity="correlation", atlas="ho", microbiome_path=None,
                             drop_age=False, drop_sex=False):
        raw_features, y, nonimg = self.load_data(
            connectivity, atlas, drop_age=drop_age, drop_sex=drop_sex
        )

        if microbiome_path is not None:
            microbiome_data = self._load_microbiome_data(microbiome_path)
        else:
            microbiome_data = self._generate_mock_microbiome_data(len(raw_features))

        return raw_features, y, nonimg, microbiome_data

    def _load_microbiome_data(self, microbiome_path):
        import pandas as pd
        try:
            df = pd.read_csv(microbiome_path, index_col=0)
            subject_IDs = Reader.get_ids()
            common_samples = []
            microbiome_features = []

            for subject_id in subject_IDs:
                if subject_id in df.index:
                    common_samples.append(subject_id)
                    microbiome_features.append(df.loc[subject_id].values.astype(np.float32))

            print(f"成功匹配 {len(common_samples)}/{len(subject_IDs)} 个样本的微生物组数据")

            self.subject_IDs = common_samples
            self.raw_features = self.raw_features[:len(common_samples)]
            self.y = self.y[:len(common_samples)]

            return np.array(microbiome_features)
        except Exception as e:
            print(f"加载微生物组数据失败: {e}")
            print("使用模拟数据代替")
            return self._generate_mock_microbiome_data(len(self.raw_features))

    def _generate_mock_microbiome_data(self, n_samples, n_features=2503):
        alpha = np.ones(n_features) * 0.1
        data = np.random.dirichlet(alpha, n_samples).astype(np.float32)
        return data

    def data_split(self, n_folds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        return list(skf.split(self.raw_features, self.y))

    def data_split_loso(self):
        """Leave-One-Site-Out: each fold leaves one site as test, rest as train."""
        site_ids = self.pd_dict["SITE_ID"]
        unique_sites = np.unique(site_ids)
        splits = []
        for site in unique_sites:
            test_ind = np.where(site_ids == site)[0]
            train_ind = np.where(site_ids != site)[0]
            if len(test_ind) == 0 or len(train_ind) == 0:
                continue
            splits.append((train_ind, test_ind))
        return splits

    def get_node_features(self, train_ind):
        node_ftr = Reader.feature_selection(self.raw_features, self.y, train_ind, self.node_ftr_dim)
        self.node_ftr = preprocess_features(node_ftr)
        return self.node_ftr

    def get_edge_inputs(self, nonimg):
        n = self.node_ftr.shape[0]
        num_edge = n * (1 + n) // 2 - n
        pd_ftr_dim = nonimg.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        edgenet_input = np.zeros([num_edge, 2 * pd_ftr_dim], dtype=np.float32)
        aff_score = np.zeros(num_edge, dtype=np.float32)
        aff_adj = Reader.get_static_affinity_adj(self.node_ftr, self.pd_dict)

        flatten_ind = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[:, flatten_ind] = [i, j]
                edgenet_input[flatten_ind] = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i][j]
                flatten_ind += 1

        assert flatten_ind == num_edge, "Error in computing edge input"
        keep_ind = np.where(aff_score > 1.1)[0]
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]
        return edge_index, edgenet_input
