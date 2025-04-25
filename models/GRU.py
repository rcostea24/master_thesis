import torch.nn as nn
import torch

class GRU(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, bidirectional=False, attention=False):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          batch_first=True, bidirectional=bidirectional)
        
        self.attention = attention
        self.bidirectional = bidirectional
        self.attn = nn.Identity()
        if attention:
            self.attn = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)

    def forward(self, x):  
        rnn_out, h_n = self.gru(x)  
        
        if self.attention:
            attn_weights = torch.softmax(self.attn(rnn_out), dim=1)  
            output = torch.sum(attn_weights * rnn_out, dim=1)
        else:
            if self.bidirectional:
                output = torch.cat((h_n[-2], h_n[-1]), dim=1)
            else:
                output = h_n[-1]
        
        return output
        
