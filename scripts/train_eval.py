import numpy as np
import torch

from AxisNet_refactor.config.opt import AxisNetOptions
from AxisNet_refactor.core.axisnet_model import AxisNetGCN, AxisNetFusion
from AxisNet_refactor.core.transformer_gcn import AxisNetTransformer, AxisNetGcnTransformer
from AxisNet_refactor.utils.metrics import accuracy, auc, prf
from AxisNet_refactor.data.loader import AxisNetDataLoader
from AxisNet_refactor.data.multimodal_loader import AxisNetMicrobiomeLoader


def build_model(opt, node_ftr_dim, nonimg_dim, microbiome_dim):
    if opt.model_type == "transformer":
        return AxisNetTransformer(
            node_ftr_dim, opt.num_classes, opt.dropout,
            edge_dropout=opt.edropout, hgc=opt.hgc, lg=opt.lg,
            edgenet_input_dim=2 * nonimg_dim,
            microbiome_dim=microbiome_dim,
            contrastive_weight=opt.contrastive_weight,
        ).to(opt.device)
    if opt.model_type == "gcn_transformer":
        return AxisNetGcnTransformer(
            node_ftr_dim, opt.num_classes, opt.dropout,
            edge_dropout=opt.edropout, hgc=opt.hgc, lg=opt.lg,
            edgenet_input_dim=2 * nonimg_dim,
            microbiome_dim=microbiome_dim,
            contrastive_weight=opt.contrastive_weight,
        ).to(opt.device)
    if opt.use_multimodal:
        return AxisNetFusion(
            node_ftr_dim, opt.num_classes, opt.dropout,
            edge_dropout=opt.edropout, hgc=opt.hgc, lg=opt.lg,
            edgenet_input_dim=2 * nonimg_dim,
            microbiome_dim=microbiome_dim,
            contrastive_weight=opt.contrastive_weight,
        ).to(opt.device)
    return AxisNetGCN(
        node_ftr_dim, opt.num_classes, opt.dropout,
        edge_dropout=opt.edropout, hgc=opt.hgc, lg=opt.lg,
        edgenet_input_dim=2 * nonimg_dim,
    ).to(opt.device)


def run_cv(opt):
    print("  Loading dataset ...")
    dl = AxisNetDataLoader()

    if opt.use_multimodal:
        print("  Loading enhanced multimodal data (fMRI-dominant + microbiome augmentation)...")
        enhanced_dl = AxisNetMicrobiomeLoader(dl)
        raw_features, y, nonimg, microbiome_data = enhanced_dl.load_multimodal(
            microbiome_path=opt.microbiome_path,
            similarity_threshold=0.8,
            top_k=opt.microbiome_top_k,
            microbiome_pca_dim=opt.microbiome_pca_dim,
            drop_age=opt.drop_age,
            drop_sex=opt.drop_sex,
        )
        print(f"  Microbiome features shape: {microbiome_data.shape}")
    else:
        print("  Loading unimodal data (fMRI only)...")
        raw_features, y, nonimg = dl.load_data(drop_age=opt.drop_age, drop_sex=opt.drop_sex)
        microbiome_data = None

    n_folds = 10
    cv_splits = dl.data_split(n_folds)

    corrects = np.zeros(n_folds, dtype=np.int32)
    accs = np.zeros(n_folds, dtype=np.float32)
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds, 3], dtype=np.float32)

    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold))
        train_ind = cv_splits[fold][0]
        test_ind = cv_splits[fold][1]

        print("  Constructing graph data...")
        node_ftr = dl.get_node_features(train_ind)
        edge_index, edgenet_input = dl.get_edge_inputs(nonimg)

        edgenet_mean = edgenet_input.mean(axis=0)
        edgenet_std = edgenet_input.std(axis=0)
        edgenet_std[edgenet_std < 1e-6] = 1.0
        edgenet_input = (edgenet_input - edgenet_mean) / edgenet_std

        if opt.use_multimodal and microbiome_data is not None:
            microbiome_dim = microbiome_data.shape[1]
        else:
            microbiome_dim = opt.microbiome_dim

        model = build_model(opt, node_ftr.shape[1], nonimg.shape[1], microbiome_dim)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        features_cuda = torch.tensor(node_ftr, dtype=torch.float32).to(opt.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(opt.device)
        labels = torch.tensor(y, dtype=torch.long).to(opt.device)

        if opt.use_multimodal and microbiome_data is not None:
            fold_microbiome = torch.tensor(microbiome_data, dtype=torch.float32).to(opt.device)
            print(f"  Microbiome data shape: {fold_microbiome.shape}")
        else:
            fold_microbiome = None

        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)

        def train():
            print("  Number of training samples %d" % len(train_ind))
            print("  Start training...\r\n")
            acc = 0
            for epoch in range(opt.num_iter):
                model.train()
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    if opt.use_multimodal and fold_microbiome is not None:
                        node_logits, edge_weights, microbiome_embed, brain_embed = model(
                            features_cuda, edge_index, edgenet_input, fold_microbiome
                        )
                        classification_loss = loss_fn(node_logits[train_ind], labels[train_ind])
                        contrastive_loss = model.contrastive_loss(
                            microbiome_embed[train_ind], brain_embed[train_ind], labels[train_ind]
                        )
                        reg_weight = opt.microbiome_reg_weight if epoch >= opt.microbiome_warmup_epochs else 0.0
                        reg_loss = model.graph_consistency_loss(
                            getattr(model, "edge_index_used", edge_index), edge_weights, fold_microbiome
                        )
                        loss = classification_loss + opt.contrastive_weight * contrastive_loss + reg_weight * reg_loss
                    else:
                        node_logits, edge_weights = model(features_cuda, edge_index, edgenet_input)
                        loss = loss_fn(node_logits[train_ind], labels[train_ind])

                    loss.backward()
                    optimizer.step()

                correct_train, acc_train = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])

                model.eval()
                with torch.set_grad_enabled(False):
                    if opt.use_multimodal and fold_microbiome is not None:
                        node_logits, _, _, _ = model(features_cuda, edge_index, edgenet_input, fold_microbiome)
                    else:
                        node_logits, _ = model(features_cuda, edge_index, edgenet_input)
                logits_test = node_logits[test_ind].detach().cpu().numpy()
                correct_test, acc_test = accuracy(logits_test, y[test_ind])
                auc_test = auc(logits_test, y[test_ind])
                prf_test = prf(logits_test, y[test_ind])

                print("Epoch: {},\tce loss: {:.5f},\ttrain acc: {:.5f}".format(epoch, loss.item(), acc_train.item()))
                if acc_test > acc and epoch > 9:
                    acc = acc_test
                    correct = correct_test
                    aucs[fold] = auc_test
                    prfs[fold] = prf_test
                    if opt.ckpt_path != "":
                        import os
                        if not os.path.exists(opt.ckpt_path):
                            os.makedirs(opt.ckpt_path)
                        torch.save(model.state_dict(), fold_model_path)

            accs[fold] = acc
            corrects[fold] = correct
            print("\r\n => Fold {} test accuacry {:.5f}".format(fold, acc))

        def evaluate():
            print("  Number of testing samples %d" % len(test_ind))
            print("  Start testing...")
            model.load_state_dict(torch.load(fold_model_path))
            model.eval()
            if opt.use_multimodal and fold_microbiome is not None:
                node_logits, _, _, _ = model(features_cuda, edge_index, edgenet_input, fold_microbiome)
            else:
                node_logits, _ = model(features_cuda, edge_index, edgenet_input)

            logits_test = node_logits[test_ind].detach().cpu().numpy()
            corrects[fold], accs[fold] = accuracy(logits_test, y[test_ind])
            aucs[fold] = auc(logits_test, y[test_ind])
            prfs[fold] = prf(logits_test, y[test_ind])
            print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))

        if opt.train == 1:
            train()
        elif opt.train == 0:
            evaluate()

    print("\r\n========================== Finish ==========================")
    n_samples = raw_features.shape[0]
    acc_nfold = np.sum(corrects) / n_samples
    acc_std = float(np.std(accs))
    auc_std = float(np.std(aucs))
    prf_std = np.std(prfs, axis=0)

    print("=> Average test accuracy in {}-fold CV: {:.5f}".format(n_folds, acc_nfold))
    print("=> Average test AUC in {}-fold CV: {:.5f}".format(n_folds, np.mean(aucs)))
    se, sp, f1 = np.mean(prfs, axis=0)
    print("=> Average test sensitivity {:.4f}, specificity {:.4f}, F1-score {:.4f}".format(se, sp, f1))
    print("=> Std (10-fold): acc {:.6f}, auc {:.6f}, se {:.6f}, sp {:.6f}, f1 {:.6f}".format(
        acc_std, auc_std, prf_std[0], prf_std[1], prf_std[2]
    ))

    return {
        "acc_mean": float(acc_nfold),
        "acc_std": acc_std,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": auc_std,
        "se_mean": float(se),
        "sp_mean": float(sp),
        "f1_mean": float(f1),
        "se_std": float(prf_std[0]),
        "sp_std": float(prf_std[1]),
        "f1_std": float(prf_std[2]),
    }


def main():
    opt = AxisNetOptions().initialize()
    run_cv(opt)


if __name__ == "__main__":
    main()
