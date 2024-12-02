import torch

from Environment.Tokens_Actions.Basic_Block.Comma_Token import Comma_Token
from Environment.Tokens_Actions.Basic_Block.Comment_Token import Comment_Token
from Environment.Tokens_Actions.Basic_Block.KeywordRepresentation_Token import (
    KeywordRepresentation_Token,
)
from Environment.Tokens_Actions.Basic_Block.Paranthesis_Token import Paranthesis_Token
from Environment.Tokens_Actions.Basic_Block.Quote_Token import Quote_Token
from Environment.Tokens_Actions.Basic_Block.Whitespace_Token import Whitespace_Token

keywords = KeywordRepresentation_Token.reserved_keywords
# operators
operators = ["op"]
# comma
comma = Comma_Token.comma_types
# comment
comments = list(Comment_Token.comment_type_mapping.values())
# hex
hex = ["hex"]
# string
string = ["str"]
# id
id = ["id"]
# number
number = ["num"]
# parenthesis
paranthesis = list(Paranthesis_Token.paranthesis_type_mapping.values())
# quotes
quotes = list(Quote_Token.quote_type_mapping.values())
# whitespace
whitespace = list(Whitespace_Token.whitespace_type_mapping.values())
# starting and ending token
start_end_token = ["sos", "eos"]

tokens_embedding = [
    *keywords,
    *operators,
    *comma,
    *comments,
    *hex,
    *string,
    *id,
    *number,
    *paranthesis,
    *quotes,
    *whitespace,
    *start_end_token,
]
token_to_idx = {c: i for i, c in enumerate(tokens_embedding)}
idx_to_token = {i: c for i, c in enumerate(tokens_embedding)}


def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask


def generate_padding_mask(lengths, max_len):
    # Create a padding mask for sequences of different lengths
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, length:] = True
    return mask
