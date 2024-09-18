
import torch
import torch.nn as nn
import drawspeech.modules.text_encoder.attentions as attentions

class Sketch2ContourPredictor(nn.Module):
    def __init__(self, pitch_embedding_dim, energy_embedding_dim, ffn_dim, n_bins, n_heads, n_layers):
        super(Sketch2ContourPredictor, self).__init__()

        pitch_min = energy_min = 0
        pitch_max = energy_max = 1
        self.pitch_bins = nn.Parameter(torch.linspace(pitch_min, pitch_max, n_bins - 1),requires_grad=False)
        self.energy_bins = nn.Parameter(torch.linspace(energy_min, energy_max, n_bins - 1),requires_grad=False)

        self.pitch_embedding = nn.Embedding(n_bins, pitch_embedding_dim)
        self.energy_embedding = nn.Embedding(n_bins, energy_embedding_dim)    

        assert pitch_embedding_dim == energy_embedding_dim
        embedding_dim = pitch_embedding_dim
        self.encoder = attentions.Encoder(embedding_dim, ffn_dim, n_heads, n_layers, kernel_size=3, p_dropout=0.1)
        self.linear_layer = nn.Linear(embedding_dim, 2)

    def get_pitch_embedding(self, x, mask):
        if mask is not None:
            x = x.masked_fill(mask == 0, 0.0)
        embedding = self.pitch_embedding(torch.bucketize(x, self.pitch_bins))
        return embedding

    def get_energy_embedding(self, x, mask):
        if mask is not None:
            x = x.masked_fill(mask == 0, 0.0)
        embedding = self.energy_embedding(torch.bucketize(x, self.energy_bins))
        return embedding

    def forward(self, x, pitch_sketch, energy_sketch, mask=None):
        '''
        x: expanded text embedding, [b, t, h]
        mask: [b, t], 1 for real data, 0 for padding
        '''

        if pitch_sketch is None and energy_sketch is None:
            return None, None

        if pitch_sketch is None:
            pitch_sketch_embedding = 0
        else:
            pitch_sketch_embedding = self.get_pitch_embedding(pitch_sketch, mask)
        
        if energy_sketch is None:
            energy_sketch_embedding = 0
        else:
            energy_sketch_embedding = self.get_energy_embedding(energy_sketch, mask)

        x = x + pitch_sketch_embedding + energy_sketch_embedding

        x = x.transpose(1, 2)  # [b, h, t]
        mask = mask.unsqueeze(1).to(x.dtype)  # [b, 1, t], 1 for real data, 0 for padding
        x = self.encoder(x * mask, mask)

        out = self.linear_layer(x.transpose(1, 2)) * mask.transpose(1, 2)
        pitch, energy = out.chunk(2, dim=-1)

        pitch = pitch.squeeze(-1)
        energy = energy.squeeze(-1)

        return pitch, energy
    

     
        
        
