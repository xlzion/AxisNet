import os
import csv
import numpy as np
import scipy.io as sio

from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from nilearn import connectome
from scipy.spatial import distance

pipeline = "cpac"
root_folder = "./data"
_data_folder = None
_phenotype_path = None
_subject_ids_path = None


def _get_data_folder():
    global _data_folder
    if _data_folder is not None:
        return _data_folder
    return os.path.join(root_folder, "ABIDE_pcp/cpac/filt_noglobal")


def _get_phenotype_path():
    global _phenotype_path
    if _phenotype_path is not None:
        return _phenotype_path
    return os.path.join(root_folder, "ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")


def _get_subject_ids_path():
    global _subject_ids_path
    if _subject_ids_path is not None:
        return _subject_ids_path
    return os.path.join(_get_data_folder(), "subject_IDs.txt")


def set_data_paths(data_folder=None, phenotype_path=None, subject_ids_path=None):
    """Set paths for ABIDE I/II. None = use default (ABIDE I)."""
    global _data_folder, _phenotype_path, _subject_ids_path
    _data_folder = data_folder
    _phenotype_path = phenotype_path
    _subject_ids_path = subject_ids_path




def fetch_filenames(subject_IDs, file_type):
    filemapping = {"func_preproc": "_func_preproc.nii.gz", "rois_ho": "_rois_ho.1D"}
    data_folder = _get_data_folder()
    filenames = []
    for i in range(len(subject_IDs)):
        try:
            file_id = get_file_id(subject_IDs[i])
            if file_id:
                pattern = os.path.join(data_folder, file_id + filemapping[file_type])
                if os.path.exists(pattern):
                    filenames.append(pattern)
                else:
                    filenames.append("N/A")
            else:
                filenames.append("N/A")
        except Exception:
            filenames.append("N/A")

    return filenames


def get_timeseries(subject_list, atlas_name):
    data_folder = _get_data_folder()
    timeseries = []
    valid_subjects = []
    for i in range(len(subject_list)):
        subject_folder = os.path.join(data_folder, subject_list[i])
        if not os.path.exists(subject_folder):
            print(f"Warning: Subject folder {subject_folder} does not exist, skipping subject {subject_list[i]}")
            continue

        ro_file = [f for f in os.listdir(subject_folder) if f.endswith("_rois_" + atlas_name + ".1D")]
        if not ro_file:
            print(f"Warning: No timeseries file found for subject {subject_list[i]}, skipping")
            continue

        fl = os.path.join(subject_folder, ro_file[0])
        print("Reading timeseries file %s" % fl)
        try:
            timeseries.append(np.loadtxt(fl, skiprows=0))
            valid_subjects.append(subject_list[i])
        except Exception as e:
            print(f"Warning: Could not load timeseries for subject {subject_list[i]}: {e}")
            continue

    return timeseries, valid_subjects


def subject_connectivity(timeseries, subject, atlas_name, kind, save=True, save_path=None):
    if save_path is None:
        save_path = _get_data_folder()
    print("Estimating %s matrix for subject %s" % (kind, subject))
    if kind in ["tangent", "partial correlation", "correlation"]:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]

    if save:
        subject_file = os.path.join(
            save_path, subject, subject + "_" + atlas_name + "_" + kind.replace(" ", "_") + ".mat"
        )
        sio.savemat(subject_file, {"connectivity": connectivity})

    return connectivity


def get_ids(num_subjects=None):
    path = _get_subject_ids_path()
    subject_IDs = np.genfromtxt(path, dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    return subject_IDs


def get_file_id(sub_id):
    with open(_get_phenotype_path()) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row["SUB_ID"] == str(sub_id):
                return row["FILE_ID"]
    return None


def get_subject_score(subject_list, score):
    scores_dict = {}
    with open(_get_phenotype_path()) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row["SUB_ID"] in subject_list:
                scores_dict[row["SUB_ID"]] = row[score]
    return scores_dict


def feature_selection(features, labels, train_ind, fnum):
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=100)
    featureX = features[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(features)
    return x_data


def site_percentage(train_ind, perc, subject_list):
    train_list = subject_list[train_ind]
    sites = get_subject_score(train_list, score="SITE_ID")
    unique = np.unique(list(sites.values())).tolist()
    site = np.array([unique.index(sites[train_list[x]]) for x in range(len(train_list))])

    labeled_indices = []
    for i in np.unique(site):
        id_in_site = np.argwhere(site == i).flatten()
        num_nodes = len(id_in_site)
        labeled_num = int(round(perc * num_nodes))
        labeled_indices.extend(train_ind[id_in_site[:labeled_num]])

    return labeled_indices


def get_networks(subject_list, kind, atlas_name="aal", variable="connectivity"):
    data_folder = _get_data_folder()
    all_networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder, subject, subject + "_" + atlas_name + "_" + kind + ".mat")
        matrix = sio.loadmat(fl)[variable]
        all_networks.append(matrix)

    idx = np.triu_indices_from(all_networks[0], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_networks = [np.arctanh(mat) for mat in all_networks]
    vec_networks = [mat[idx] for mat in norm_networks]
    matrix = np.vstack(vec_networks)
    return matrix


def create_affinity_graph_from_scores(scores, pd_dict):
    num_nodes = len(pd_dict[scores[0]])
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = pd_dict[l]
        if l in ["AGE_AT_SCAN", "FIQ"]:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:
                        pass
        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[k] == label_dict[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1
    return graph


def get_static_affinity_adj(features, pd_dict):
    pd_affinity = create_affinity_graph_from_scores(["SEX", "SITE_ID"], pd_dict)
    distv = distance.pdist(features, metric="correlation")
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    feature_sim = np.exp(-dist ** 2 / (2 * sigma ** 2))
    adj = pd_affinity * feature_sim
    return adj
