import os
import argparse
import torch
from torch_geometric.loader import DataLoader
from sklearn import metrics
from model import GPS
from data import ProDataset

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "ckpt/"


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for Evaluation Gated-GPS")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--esm_out", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--pe_dim", type=int, default=32)
    parser.add_argument("--pe_ratio", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attn_dropout", type=float, default=0.7)
    parser.add_argument("--act", type=str, default="ReLU")
    parser.add_argument("--weight", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.36)
    parser.add_argument("--beta", type=float, default=0.97)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--esm_name", type=str, default=None)
    parser.add_argument("--esm_model_path", type=str, default=None)
    args = parser.parse_args()
    return args

def evaluate(use_esm, model, data_loader):
    model.eval()
    epoch_loss = 0.0

    valid_pred = []
    valid_true = []

    for data in data_loader:
        with torch.no_grad():
            data = data.to(device)
            
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            pe = data.pe
            batch = data.batch
            y_true = data.y
            
            if use_esm:
                esm = data.esm_feat
                y_pred = model(x, edge_index, edge_attr, pe, batch, esm)
            else:
                y_pred = model(x, edge_index, edge_attr, pe, batch)

            # calculate loss
            loss = model.criterion(y_pred, y_true)
            
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)

            epoch_loss += loss.item()         

    epoch_loss_avg = epoch_loss / len(data_loader)
    return epoch_loss_avg, valid_true, valid_pred

def analysis(y_true, y_pred, best_threshold = None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_pred': binary_pred,
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results


def test(args, test_loader):
    model = GPS(
        channels=args.hidden_dim,
        heads=args.num_heads,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        act=args.act,
        pe_dim=args.pe_dim,
        pe_ratio=args.pe_ratio,
        esm_out=args.esm_out,
        num_layers=args.num_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        weight=args.weight,
        alpha=args.alpha,
        beta=args.beta,
        use_esm=True,
        esm_dim=args.esm_dim,
    ).to(device) 

    for model_name in sorted(os.listdir(model_path)):
        print(model_name)
        model.load_state_dict(torch.load(model_path + model_name, map_location='cuda:0'))

        epoch_loss_test_avg, test_true, test_pred = evaluate(True, model, test_loader)
        result_test = analysis(test_true, test_pred)
        
        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test mcc: ", result_test['mcc'])
        print("Threshold: ", result_test['threshold'])

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_args()
    print(args)    
    set_seed(args.seed)
    
    if args.esm_name == "esm2_t6_8M":
        args.esm_dim = 320
        args.esm_layers = 6
    elif args.esm_name == "esm2_t12_35M":
        args.esm_dim = 480
        args.esm_layers = 12
    elif args.esm_name == "esm2_t30_150M":
        args.esm_dim = 640
        args.esm_layers = 30
    elif args.esm_name == "esm2_t33_650M":
        args.esm_dim = 1280
        args.esm_layers = 33
    elif args.esm_name == "esm2_t36_3B":
        args.esm_dim = 2560
        args.esm_layers = 36
    elif args.esm_name == "esm2_t48_15B":
        args.esm_dim = 5120
        args.esm_layers = 48
    elif args.esm_name == "esm1b_t33_650M":
        args.esm_dim = 1280
        args.esm_layers = 33
    elif args.esm_name == "esm1v_t33_650M":
        args.esm_dim = 1280
        args.esm_layers = 33
    else: # by default, use esm2_t33_650M
        args.esm_dim = 1280
        args.esm_layers = 33

    print("Creating Test set Test60 ...")
    test_set60 = ProDataset(
        pe_dim=args.pe_dim,
        dataset_name="Test60",
        train=False,
        esm_name=args.esm_name,
        esm_layers=args.esm_layers,
        esm_model_path=args.esm_model_path,
        feature_path="datasets/feature/feature_test_60/"
    )
    
    print("Creating Test set Test315 ...")
    test_set315 = ProDataset(
        pe_dim=args.pe_dim,
        dataset_name="Test315",
        train=False,
        esm_name=args.esm_name,
        esm_layers=args.esm_layers,
        esm_model_path=args.esm_model_path,
        feature_path="datasets/feature/feature_test_315/"
    )

    print("Creating Test set Btest31...")
    test_b31 = ProDataset(
        pe_dim=args.pe_dim,
        dataset_name="Btest31",
        train=False,
        esm_name=args.esm_name,
        esm_layers=args.esm_layers,
        esm_model_path=args.esm_model_path,
        feature_path="datasets/feature/feature_test_60/"
    )

    print("Creating Test set UBtest31...")
    test_ub31 = ProDataset(
        pe_dim=args.pe_dim,
        dataset_name="UBtest31",
        train=False,
        esm_name=args.esm_name,
        esm_layers=args.esm_layers,
        esm_model_path=args.esm_model_path,
        feature_path="datasets/feature/feature_ubtest_31/"
    )

    print("len of test60: ", len(test_set60))
    print("len of test315: ", len(test_set315))
    print("len of btest31: ", len(test_b31))
    print("len of ubtest31: ", len(test_ub31))
    
    
    print("Evaluating on Test60 ...")
    test_loader = DataLoader(test_set60, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    test(args, test_loader)
    
    print("Evaluating on Test315 ...")
    test_loader = DataLoader(test_set315, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    test(args, test_loader)

    print("Evaluating on Btest31 ...")
    test_loader = DataLoader(test_b31, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    test(args, test_loader)

    print("Evaluating on UBtest31 ...")
    test_loader = DataLoader(test_ub31, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    test(args, test_loader)