import torch.nn as nn
import torch

class NAVAR(nn.Module):
    def __init__(self, num_nodes, num_hidden, maxlags, hidden_layers=1, dropout=0):
        """
        Neural Additive Vector AutoRegression (NAVAR) model
        Args:
            num_nodes: int
                The number of time series (N)
            num_hidden: int
                Number of hidden units per layer
            maxlags: int
                Maximum number of time lags considered (K)
            hidden_layers: int
                Number of hidden layers
            dropout:
                Dropout probability of units in hidden layers
        """
        super(NAVAR, self).__init__()

        self.num_nodes = num_nodes
        self.num_hidden = num_hidden
        self.first_hidden_layer = nn.Conv1d(num_nodes, num_hidden * num_nodes, kernel_size=maxlags,
                                                  groups=num_nodes)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_layer_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        for k in range(hidden_layers - 1):
            self.hidden_layer_list.append(
                nn.Conv1d(num_nodes, num_hidden * num_nodes, kernel_size=num_hidden, groups=num_nodes))
            self.dropout_list.append(nn.Dropout(p=dropout))
        self.contributions = nn.Conv1d(num_nodes, num_nodes * num_nodes, kernel_size=num_hidden, groups=num_nodes)
        self.biases = nn.Parameter(torch.ones(1, num_nodes) * 0.0001)

    def forward(self, x):
        hidden = self.first_hidden_layer(x).clamp(min=0).view([-1, self.num_nodes, self.num_hidden])
        hidden = self.dropout(hidden)
        for i in range(len(self.hidden_layer_list)):
            hidden = self.hidden_layer_list[i](hidden).clamp(min=0).view([-1, self.num_nodes, self.num_hidden])
            hidden = self.dropout_list[i](hidden)

        contributions = self.contributions(hidden)
        contributions = contributions.view([-1, self.num_nodes, self.num_nodes, 1])
        predictions = torch.sum(contributions, dim=1).squeeze() + self.biases
        contributions = contributions.view([-1, self.num_nodes*self.num_nodes, 1]).squeeze()
        return predictions, contributions
