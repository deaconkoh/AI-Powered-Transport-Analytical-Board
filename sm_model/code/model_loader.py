"""
Model loader for trained traffic prediction models
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import os
import json
from typing import Dict, Any

# Define model architectures (same as your training)
class RobustSTGNN(nn.Module):
    def __init__(self, num_nodes, in_feats, hidden_dim=64, seq_len=12):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.lstm = nn.LSTM(in_feats, hidden_dim, batch_first=True, num_layers=1)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.output_layer = nn.Linear(hidden_dim, num_nodes)

    def forward(self, x, edge_index=None, edge_attr=None):
        batch_size, seq_len, _ = x.shape
        lstm_out, (h_n, c_n) = self.lstm(x)
        temporal_features = lstm_out[:, -1, :]
        spatial_features = self.spatial_mlp(temporal_features)
        out = self.output_layer(spatial_features)
        return out

class SimpleSTGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_dim=64, seq_len=12):
        super(SimpleSTGCN, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.gcn = GCNConv(in_channels, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_weight):
        batch_size, seq_len, num_nodes, in_channels = x.shape
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :].reshape(-1, in_channels)
            x_gcn = torch.relu(self.gcn(x_t, edge_index, edge_weight))
            gcn_outputs.append(x_gcn.view(batch_size, num_nodes, -1))
        
        gcn_stacked = torch.stack(gcn_outputs, dim=1)
        node_predictions = []
        for node_idx in range(num_nodes):
            node_features = gcn_stacked[:, :, node_idx, :]
            lstm_out, _ = self.lstm(node_features)
            node_pred = self.fc(self.dropout(lstm_out[:, -1, :]))
            node_predictions.append(node_pred)
        
        predictions = torch.cat(node_predictions, dim=1)
        return predictions

class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes, in_feats, seq_len=12, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.in_feats = in_feats
        
        self.temp_conv1 = nn.Conv2d(in_feats, hidden_dim, kernel_size=(1, 3), padding=(0, 1))
        self.temp_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1))
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.output_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        batch_size = x.shape[0]
        x = x.permute(0, 3, 2, 1)
        x = self.relu(self.temp_conv1(x))
        x = self.dropout(x)
        x = self.relu(self.temp_conv2(x))
        x = self.dropout(x)
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(batch_size * self.seq_len, self.num_nodes, -1)
        x = self.relu(self.gcn1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.gcn2(x, edge_index))
        x = x.reshape(batch_size, self.seq_len, self.num_nodes, -1)
        x = x.permute(0, 3, 2, 1)
        x = self.output_conv(x)
        output = x[:, :, :, -1].squeeze(1)
        return output

class TrafficModelLoader:
    def __init__(self, model_dir: str = None):
        # Get the project root directory (where server.py is located)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.model_dir = model_dir or os.path.join(project_root, "models")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.loaded = False
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        print(f"Model directory: {self.model_dir}")
        
    def load_models(self):
        """Load all three trained models"""
        if self.loaded:
            return self.models
            
        try:
            print(f"Looking for models in: {self.model_dir}")
            
            # Check if model files exist
            model_files = ['custom_stgnn_best.pth', 'stgcn_finetuned_best.pth', 'graphwavenet_finetuned_best.pth']
            missing_files = []
            
            for model_file in model_files:
                model_path = os.path.join(self.model_dir, model_file)
                if os.path.exists(model_path):
                    print(f"Found: {model_file}")
                else:
                    print(f"Missing: {model_file} at {model_path}")
                    missing_files.append(model_file)
            
            if missing_files:
                print(f"Missing model files: {missing_files}")
                print("Please make sure your .pth files are in the models/ directory")
                return {}
            
            # Use the correct parameters that match your trained models
            model_params = {
                'custom': {'num_nodes': 100, 'in_feats': 108, 'hidden_dim': 64, 'seq_len': 12},
                'stgcn': {'num_nodes': 100, 'in_channels': 9, 'hidden_dim': 64, 'seq_len': 12},
                'graphwavenet': {'num_nodes': 100, 'in_feats': 9, 'hidden_dim': 64, 'seq_len': 12}
            }
            
            # Load models with error handling for each one
            loaded_models = {}
            
            # Load custom model
            try:
                print("Loading Custom STGNN...")
                custom_model = RobustSTGNN(**model_params['custom'])
                custom_model.load_state_dict(
                    torch.load(
                        os.path.join(self.model_dir, 'custom_stgnn_best.pth'), 
                        map_location=self.device
                    )
                )
                custom_model.eval()
                loaded_models['custom'] = custom_model
                print("Custom STGNN loaded successfully!")
            except Exception as e:
                print(f"Failed to load Custom STGNN: {e}")
            
            # Load STGCN model
            try:
                print("Loading STGCN...")
                stgcn_model = SimpleSTGCN(**model_params['stgcn'])
                stgcn_model.load_state_dict(
                    torch.load(
                        os.path.join(self.model_dir, 'stgcn_finetuned_best.pth'),
                        map_location=self.device
                    )
                )
                stgcn_model.eval()
                loaded_models['stgcn'] = stgcn_model
                print("STGCN loaded successfully!")
            except Exception as e:
                print(f"Failed to load STGCN: {e}")
            
            # Load Graph WaveNet model
            try:
                print("Loading Graph WaveNet...")
                graphwavenet_model = GraphWaveNet(**model_params['graphwavenet'])
                graphwavenet_model.load_state_dict(
                    torch.load(
                        os.path.join(self.model_dir, 'graphwavenet_finetuned_best.pth'),
                        map_location=self.device
                    )
                )
                graphwavenet_model.eval()
                loaded_models['graphwavenet'] = graphwavenet_model
                print("Graph WaveNet loaded successfully!")
            except Exception as e:
                print(f"Failed to load Graph WaveNet: {e}")
            
            self.models = loaded_models
            self.loaded = True
            
            print(f"Successfully loaded {len(self.models)} out of 3 models")
            
            # If no models loaded, return empty dict
            if not self.models:
                print("No models were successfully loaded")
                return {}
                
            return self.models
            
        except Exception as e:
            print(f"Error during model loading: {e}")
            import traceback
            traceback.print_exc()
            self.models = {}
            return self.models

# Global instance
_model_loader = None

def get_model_loader():
    global _model_loader
    if _model_loader is None:
        _model_loader = TrafficModelLoader()
    return _model_loader