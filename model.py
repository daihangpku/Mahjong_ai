import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        
        self.encoder_layer = TransformerEncoderLayer(dim_feedforward, nhead, dim_feedforward, batch_first=True)
        self.encoder = TransformerEncoder(self.encoder_layer, num_encoder_layers)
        
        self.decoder_layer = TransformerDecoderLayer(dim_feedforward, nhead, dim_feedforward, batch_first=True)
        self.decoder = TransformerDecoder(self.decoder_layer, num_decoder_layers)
        
        self.fc1 = nn.Linear(dim_feedforward, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU(inplace=True)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
    def forward(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        action_mask = input_dict["obs"]["action_mask"].float()

        # Flattening the observation input to feed into the transformer
        obs = obs.view(obs.size(0), -1)  # Flatten to (batch_size, input_dim)
        x = self.embedding(obs)  # Embed the input

        x = x.unsqueeze(1)  # Add a dummy sequence dimension for transformer input
        memory = self.encoder(x)

        # Decoder input can be zeros or some initial state
        decoder_input = torch.zeros_like(memory)
        x = self.decoder(decoder_input, memory)
        x = x.squeeze(1)  # Remove the dummy sequence dimension

        x = self.fc1(x)
        x = self.relu(x)
        action_logits = self.fc2(x)
        
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask

# Example usage:
# Define the model with appropriate input_dim and other parameters
input_dim = 5292  # This should match the number of features in your input data
model = TransformerModel(input_dim=input_dim, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512, num_classes=10)

# Example input_dict
input_dict = {
    "obs": {
        "observation": torch.randn(128, 5292),  # Example batch of observations
        "action_mask": torch.ones(128, 10)  # Example action mask
    },
    "is_training": True
}

# Forward pass
logits = model(input_dict)
print(logits.shape)
