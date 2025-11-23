import os
import esm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from tqdm import tqdm

# model parameters
MAP_CUTOFF = 14
DIST_NORM = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProDataset(Dataset):
    def __init__(
            self,
            pe_dim, 
            radius=MAP_CUTOFF, 
            dist=DIST_NORM, 
            data_path="datasets/",
            dataset_name="Train2771",
            feature_path="datasets/feature/",
            train=True,
            esm_name=None,
            esm_layers=None,
            esm_model_path=None,
        ):
        self.transform = T.AddRandomWalkPE(walk_length=pe_dim, attr_name="pe")
        
        self.radius = radius
        self.dist = dist
        self.feature_path = feature_path
        self.esm_name = esm_name
        self.esm_layers = esm_layers
        self.esm_model_path = esm_model_path

        file_name = f"{dataset_name}_graph_list_esm.pkl"
        if esm_name:
            file_name = file_name.replace("esm.pkl", esm_name + ".pkl")
        if not os.path.exists(file_name):
            self.prepare_data(data_path, dataset_name, feature_path, train)
        
        print("Loading Graph List ...")
        with open(file_name, "rb") as f:
            Graph_list = pickle.load(f) 
            
        print("Transforming Graph List ...") 
        self.Graph_list = []
        for graph in tqdm(Graph_list):
            assert graph.num_nodes == len(graph.y)
            self.Graph_list.append(self.transform(graph))

    def __getitem__(self, index):
        return self.Graph_list[index]

    def __len__(self):
        return len(self.Graph_list)  
    
    def prepare_data(self, data_path, dataset_name, feature_path, train):
        if train: # Training set
            if dataset_name == "Train2771":
                with open(data_path + "Train_2771.pkl", "rb") as f:
                    data = pickle.load(f)
            elif dataset_name == "Train334":
                with open(data_path + "Train_334.pkl", "rb") as f:
                    data = pickle.load(f)
        else: # Test set
            if dataset_name == "Test60":
                with open(data_path + "Test_60.pkl", "rb") as f:
                    data = pickle.load(f)
            elif dataset_name == "Test315":
                with open(data_path + "Test_315.pkl", "rb") as f:
                    data = pickle.load(f)
            elif dataset_name == "UBtest31":
                with open(data_path + "UBtest_31.pkl", "rb") as f:
                    data = pickle.load(f)
            elif dataset_name == "Btest31":
                with open(data_path + "Test_60.pkl", "rb") as f:
                    test60 = pickle.load(f)
                data = {}
                with open(data_path + "bound_unbound_mapping.txt", "r") as f:
                    lines = f.readlines()[1:]
                for line in lines:
                    bound_ID, unbound_ID, _ = line.strip().split()
                    data[bound_ID] = test60[bound_ID]
        
        # esm model
        if self.esm_model_path and os.path.exists(self.esm_model_path):
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(self.esm_model_path)
        else:
            if self.esm_name == "esm2_t6_8M":
                model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            elif self.esm_name == "esm2_t12_35M":
                model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            elif self.esm_name == "esm2_t30_150M":
                model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            elif self.esm_name == "esm2_t33_650M":
                model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            elif self.esm_name == "esm2_t36_3B":
                model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            elif self.esm_name == "esm2_t48_15B":
                model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
            elif self.esm_name == "esm1b_t33_650M":
                model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            elif self.esm_name == "esm1v_t33_650M":
                model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_[1 - 5]()
            else: # by default, use esm2_t33_650M
                model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        batch_converter = alphabet.get_batch_converter()
        model.to(device)
        model.eval()
        
        self.Graph_list = []
        self.labels = []
        
        for ID in tqdm(data, desc="Processing data"):
            item = data[ID]
            ID = ID[:4].lower() + ID[-1]
            sequence_name = ID
            sequence = item[0]
            label = item[1]
            self.labels.append(label)

            with open(feature_path + "psepos/" + ID + '_psepos_SC.pkl', "rb") as f:
                pos = pickle.load(f)

                reference_res_psepos = pos[0]
                pos = pos - reference_res_psepos
                pos = torch.from_numpy(pos)

            blosum_feature = np.load(feature_path + "blosum/" + ID + '.npy')
            dssp_feature = np.load(feature_path + "dssp/" + ID + '.npy')
            node_features = np.concatenate([blosum_feature, dssp_feature], axis = 1).astype(np.float32)
            node_features = torch.from_numpy(node_features)

            if node_features.size(0) != len(label):
                print(f"Sequence {ID} didn't match num nodes == len(label)!")
                continue
            
            # esm data
            prodata = [("data", sequence)]

            batch_labels, batch_strs, batch_tokens = batch_converter(prodata)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[self.esm_layers], return_contacts=True)
            token_representations = results["representations"][self.esm_layers]
            tk_len = token_representations.shape[1]
            esm_emb = token_representations[:,1:tk_len-1,:]
            esm_emb = esm_emb.squeeze().detach().to('cpu')
        
            res_atom_features = np.load(feature_path + "resAF/" + ID + '.npy')
            res_atom_features = torch.from_numpy(res_atom_features.astype(np.float32))
            node_features = torch.cat([node_features, res_atom_features], dim=-1)
            one_hot_feature = self.get_onehot(sequence)
            node_features = torch.cat([node_features, one_hot_feature], dim=-1)
        
            node_features = torch.cat([node_features, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)

            radius_index_list = self.cal_edges(sequence_name)
            edge_feat = self.cal_edge_attr(radius_index_list, pos)
            edge_feat = np.transpose(edge_feat, (1, 2, 0))
            edge_feat = edge_feat.squeeze(1)
            
            edge_feat = torch.from_numpy(edge_feat).type(torch.FloatTensor)
            
            edge_index = torch.tensor(radius_index_list)
            
            Graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_feat, y=torch.tensor(label), esm_feat=esm_emb)
        
            self.Graph_list.append(Graph)
        
        with open(f'{dataset_name}_graph_list_esm.pkl', 'wb') as f:
            pickle.dump(self.Graph_list, f)    

    
    def get_onehot(self, seq):
        aLetters_ = np.array(
            ['H', 'D', 'R', 'F', 'A', 'C', 'G', 'Q', 'E', 'K', 'L', 'M', 'N', 'S', 'Y', 'T', 'I', 'W', 'P', 'V', 'U', 'O',
            'B', 'J', 'Z', 'X'])
        index_ = []
        for i in seq:
            for j in range(len(aLetters_)):
                if aLetters_[j] == i:
                    index_.append(j)
        index_ = torch.tensor(index_)
        one_hot = torch.squeeze(F.one_hot(index_, 26).float())
        return one_hot

    def cal_edges(self, sequence_name):  # to get the index of the edges
        dist_matrix = np.load(self.feature_path + "dismap/" + sequence_name + ".npy")
        mask = ((dist_matrix >= 0) * (dist_matrix <= self.radius))
        adjacency_matrix = mask.astype(np.int32)
        radius_index_list = np.where(adjacency_matrix == 1)
        radius_index_list = [list(nodes) for nodes in radius_index_list]
        return radius_index_list

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2,keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)

        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        radius_attr_list = np.array([distance, cos])
        return radius_attr_list

def get_train_validation_data_loaders(train_dataset, batch_size, train_idx, valid_idx, num_workers):
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                num_workers=num_workers, drop_last=False)

    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler,
                                num_workers=num_workers, drop_last=False)

    return train_loader, valid_loader