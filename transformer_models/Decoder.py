import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import generate_padding_mask, generate_square_subsequent_mask


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        output_size,
        max_length,
        num_heads=1,
        num_layers=1,
        dropout_p=0.1,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.zeros(max_length, hidden_size))

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, decoder_input, encoder_outputs, tgt_lengths, memory_lengths):
        embedded = (
            self.embedding(decoder_input)
            + self.positional_encoding[: decoder_input.size(0), :]
        )
        embedded = self.dropout(embedded).permute(1, 0, 2)

        tgt_mask = generate_square_subsequent_mask(embedded.size(0))
        memory_mask = generate_padding_mask(memory_lengths, encoder_outputs.size(0))
        tgt_key_padding_mask = generate_padding_mask(tgt_lengths, embedded.size(0))

        decoder_outputs = self.transformer_decoder(
            tgt=embedded,
            memory=encoder_outputs,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        output = F.log_softmax(self.out(decoder_outputs), dim=-1)
        return output

    def decode(self, decoder_input, encoder_output, length):
        result = []
        input_lengths = [length] * length.size(1)  # Assuming batch input
        tgt_lengths = [length] * length.size(1)
        for _ in range(length):
            decoder_output = self(
                decoder_input,
                encoder_output,
                tgt_lengths=tgt_lengths,
                memory_lengths=input_lengths,
            )
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            result.append(topi.cpu().detach().numpy()[0][0])
        return result
