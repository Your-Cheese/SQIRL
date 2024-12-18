import os

import torch

from Environment.SQL.SQL import SQL
from Environment.Tokens_Actions.Basic_Block.Comma_Token import Comma_Token
from Environment.Tokens_Actions.Basic_Block.Comment_Token import Comment_Token
from Environment.Tokens_Actions.Basic_Block.KeywordRepresentation_Token import (
    KeywordRepresentation_Token,
)
from Environment.Tokens_Actions.Basic_Block.Paranthesis_Token import Paranthesis_Token
from Environment.Tokens_Actions.Basic_Block.Quote_Token import Quote_Token
from Environment.Tokens_Actions.Basic_Block.Whitespace_Token import Whitespace_Token
from transformer_models.Encoder import (
    Encoder,
)


class SQL_Representation:
    # get all different possible tokens and join them
    # keywords
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

    def __init__(
        self,
        db_type,
        input_size,
        output_size,
        hidden_size,
        max_length,
        acc_threshold=0.80,
    ) -> None:
        self.encoder = Encoder(input_size, hidden_size, max_length)
        self.encoder.load_state_dict(
            torch.load(
                os.path.join(
                    "transformer_models",
                    "SQL_encoder.pt",
                ),
                weights_only=True,
            )
        )
        # self.decoder = Decoder(hidden_size, output_size, max_length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        # load trained models
        # self.encoder = torch.load(
        #     os.path.join(
        #         # "RL_Agent",
        #         # "State_Representation",
        #         # "Generic_Syntax_State_Representation",
        #         # "model",
        #         # "encoder.model",
        #     ),
        #     map_location=self.device,
        # )
        # self.decoder = torch.load(
        #     os.path.join(
        #         "RL_Agent",
        #         "State_Representation",
        #         "Generic_Syntax_State_Representation",
        #         "model",
        #         "decoder.model",
        #     ),
        #     map_location=self.device,
        # )

        self.acc_threshold = acc_threshold

    def embedding(sql: SQL):
        result = []

        for current_token in sql.get_tokens().flat_idx_tokens_list():
            result.append(
                SQL_Representation.token_to_idx[str(current_token).casefold()]
            )
        return result

    def generate_representation(self, sql: SQL, token: str, type=2):
        # convert sql to generic and trim
        generic_sql = sql.get_generic_statment()

        # embbed data
        embedded_data = SQL_Representation.embedding(generic_sql)

        # convert to tensor input
        tensored_input = torch.tensor(
            embedded_data, dtype=torch.long, device=self.device
        ).view(-1, 1)

        # get length of input
        input_length = tensored_input.size(0)

        # encode
        output = self.encoder.encode(tensored_input, input_length)
        features = torch.sum(output, dim=0)
        return features
