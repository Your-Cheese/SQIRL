import torch
import torch.nn as nn

from transformer_models.utils import generate_padding_mask


class Encoder(
    nn.Module,
):
    def __init__(self, input_size, hidden_size, max_length, num_heads=1, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.zeros(max_length, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, input_tensor, lengths):
        embedded = self.embedding(input_tensor) + self.positional_encoding[
            : input_tensor.size(0), :
        ].unsqueeze(1)
        # embedded = embedded.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, hidden_size]
        mask = generate_padding_mask(lengths, embedded.size(0)).to(embedded.device)
        encoder_outputs = self.transformer_encoder(embedded, src_key_padding_mask=mask)
        return encoder_outputs

    def encode(self, input_tensor, length):
        input_lengths = [length] * input_tensor.size(1)
        encoder_outputs = self(input_tensor, input_lengths)

        return encoder_outputs
