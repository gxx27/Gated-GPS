import os
import time
import argparse
import torch
from torch_geometric.loader import DataLoader
from sklearn import metrics
from sklearn.model_selection import KFold
import wandb
from model import GPS
from data import ProDataset, get_train_validation_data_loaders

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training Gated-GPS")
    # Training
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--esm_out", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--pe_dim", type=int, default=30)
    parser.add_argument("--pe_ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn_dropout", type=float, default=0.4)
    parser.add_argument("--act", type=str, default="relu")
    parser.add_argument("--weight", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--use_esm", action="store_true")
    
    # Others
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--esm_name", type=str, default=None)
    parser.add_argument("--esm_model_path", type=str, default=None)
    args = parser.parse_args()
    return args


def train_one_epoch(use_esm, model, data_loader):
    model.train()
    epoch_loss_train = 0.0
    
    for data in data_loader:
        model.optimizer.zero_grad()
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

        def closure():
            loss = model.criterion(model(x, edge_index, edge_attr, pe, batch, esm), y_true)
            loss.backward()
            return loss
        
        loss = model.criterion(y_pred, y_true)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model.optimizer.step(closure)

        epoch_loss_train += loss.item()
    epoch_loss_train_avg = epoch_loss_train / len(data_loader)
    wandb.log({'total_train_loss':epoch_loss_train_avg})
    return epoch_loss_train_avg


def evaluate(use_esm, model, data_loader, valid=False, test=False):
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
    if valid:
        wandb.log({'total_valid_loss':epoch_loss_avg})
    if test:
        wandb.log({'total_test_loss':epoch_loss_avg})
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


def train(args, model, train_loader, valid_loader, test_loader, number):
    best_epoch = 0
    best_val_aupr = 0
    for epoch in range(args.epochs):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        epoch_loss_train_avg = train_one_epoch(args.use_esm, model, train_loader)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred = evaluate(args.use_esm, model, train_loader)
        result_train = analysis(train_true, train_pred, 0.5)
        print(f"Train loss:{epoch_loss_train_avg}, Train binary acc:{result_train['binary_acc']}, Train AUC:{result_train['AUC']}, Train AUPRC:{result_train['AUPRC']}")

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred = evaluate(args.use_esm, model, valid_loader, valid=True)
        result_valid = analysis(valid_true, valid_pred, 0.5)
        print(f"Valid loss:{epoch_loss_valid_avg}, Valid binary acc:{result_valid['binary_acc']}, Valid precision:{result_valid['precision']}, Valid recall:{result_valid['recall']}, Valid f1:{result_valid['f1']}, Valid AUC:{result_valid['AUC']}, Valid AUPRC:{result_valid['AUPRC']}, Valid mcc:{result_valid['mcc']}")

        print("========== Evaluate Test set ==========")
        epoch_loss_test_avg, test_true, test_pred = evaluate(args.use_esm, model, test_loader, test=True)
        result_test = analysis(test_true, test_pred, 0.5)
        print(f"Test loss:{epoch_loss_test_avg}, Test binary acc:{result_test['binary_acc']}, Test precision:{result_test['precision']}, Test recall:{result_test['recall']}, Test f1:{result_test['f1']}, Test AUC:{result_test['AUC']}, Test AUPRC:{result_test['AUPRC']}, Test mcc:{result_test['mcc']}")

        if best_val_aupr < result_valid['AUPRC']:
            best_epoch = epoch + 1
            best_val_aupr = result_valid['AUPRC']
            torch.save(model.state_dict(), os.path.join(args.save_path, f'Fold_{str(number)}_best_model.pth'))

        model.main_scheduler.step(result_valid['AUPRC'])

    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val AUPRC: {best_val_aupr}")

def test(args, test_loader, number):
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
        use_esm=args.use_esm,
        esm_dim=args.esm_dim,
    ).to(device) 

    model.load_state_dict(torch.load(os.path.join(args.save_path, f'Fold_{str(number)}_best_model.pth'), map_location='cuda:0'))

    epoch_loss_test_avg, test_true, test_pred = evaluate(args.use_esm, model, test_loader)
    result_test = analysis(test_true, test_pred)
    
    return result_test

def cross_validation(args, trainset, test_set60, test_set315, test_b31, test_ub31, number=5):
    kfold = KFold(n_splits=number, shuffle=True, random_state=args.seed)
    test60_loader = DataLoader(test_set60, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    test315_loader = DataLoader(test_set315, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    testb31_loader = DataLoader(test_b31, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    testub31_loader = DataLoader(test_ub31, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    for i, (train_idx, valid_idx) in enumerate(kfold.split(trainset)):
        train_loader, valid_loader = get_train_validation_data_loaders(trainset, args.batch_size, train_idx, valid_idx, num_workers=0)

        channel = args.hidden_dim
        model = GPS(
            channels=channel,
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
            use_esm=args.use_esm,
            esm_dim=args.esm_dim,
        ).to(device)
        
        model.get_optimizer_scheduler()  

        train(args, model, train_loader, valid_loader, test60_loader, i+1)
        test60_result = test(args, test60_loader, i+1)
        test315_result = test(args, test315_loader, i+1)
        testb31_result = test(args, testb31_loader, i+1)
        testub31_result = test(args, testub31_loader, i+1)
        
        print("========== Result of Test60 ==========")
        for metric in sorted(test60_result):
            print(f"{metric}: {test60_result[metric]}")
        print("========== Result of Test315 ==========")
        for metric in sorted(test315_result):
            print(f"{metric}: {test315_result[metric]}")
        print("========== Result of Btest31 ==========")
        for metric in sorted(testb31_result):
            print(f"{metric}: {testb31_result[metric]}")        
        print("========== Result of UBtest31 ==========")
        for metric in sorted(testub31_result):
            print(f"{metric}: {testub31_result[metric]}")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_args()
    wandb.init(project="Gated-GPS")
    if args.esm_name:
        wandb.run.name = "Gated-GPS run " + args.esm_name
    else:
        wandb.run.name = "Gated-GPS run"
        
    print(args)
    set_seed(args.seed)
    
    os.makedirs(args.save_path, exist_ok=True)

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
    
    print("Creating Training set ...")
    start = time.time()
    train_set = ProDataset(
        pe_dim=args.pe_dim,
        esm_name=args.esm_name,
        esm_layers=args.esm_layers,
        esm_model_path=args.esm_model_path,
        feature_path="datasets/feature/feature_train_2771/"
    )
    end = time.time()
    print("Creating datasets costs:", end-start)
    
    print("Creating Test set Test_60 ...")
    test_set60 = ProDataset(
        pe_dim=args.pe_dim,
        dataset_name="Test60",
        train=False,
        esm_name=args.esm_name,
        esm_layers=args.esm_layers,
        esm_model_path=args.esm_model_path,
        feature_path="datasets/feature/feature_test_60/"
    )
    
    print("Creating Test set Test_315 ...")
    test_set315 = ProDataset(
        pe_dim=args.pe_dim,
        dataset_name="Test315",
        train=False,
        esm_name=args.esm_name,
        esm_layers=args.esm_layers,
        esm_model_path=args.esm_model_path,
        feature_path="datasets/feature/feature_test_315/"
    )
    
    print("Creating Test set Btest_31...")
    test_b31 = ProDataset(
        pe_dim=args.pe_dim,
        dataset_name="Btest31",
        train=False,
        esm_name=args.esm_name,
        esm_layers=args.esm_layers,
        esm_model_path=args.esm_model_path,
        feature_path="datasets/feature/feature_test_60/"
    )

    print("Creating Test set UBtest_31...")
    test_ub31 = ProDataset(
        pe_dim=args.pe_dim,
        dataset_name="UBtest31",
        train=False,
        esm_name=args.esm_name,
        esm_layers=args.esm_layers,
        esm_model_path=args.esm_model_path,
        feature_path="datasets/feature/feature_ubtest_31/"
    )
        
    print("len of train:", len(train_set))
    print("len of test60: ", len(test_set60))
    print("len of test315: ", len(test_set315))
    print("len of btest31: ", len(test_b31))
    print("len of ubtest31: ", len(test_ub31))
    
    cross_validation(args, train_set, test_set60, test_set315, test_b31, test_ub31)
