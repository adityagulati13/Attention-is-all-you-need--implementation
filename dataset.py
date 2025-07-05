import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt,src_lang,tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        # generate the tokens for --> SOS, EOS, PAD
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        # getting the src_tgt_pair
        src_target_pair = self.ds[idx]
        # getting src text
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]
        # Transform text to tokens, tokens to text  --> use .encode()
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        enc_input_tokens = enc_input_tokens[: self.seq_len - 2]  # [SOS] + tokens + [EOS]
        dec_input_tokens = dec_input_tokens[: self.seq_len - 1]  # [SOS] + tokens
        # adding sos , eos, and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) -2
        # we will only  add </s> to the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) -1
        # ensuring that the number of padding token is not negative

        # adding <s>, and </s> to the encoder input
        encoder_input = torch.cat(
            [self.sos_token,
             torch.tensor(enc_input_tokens, dtype= torch.int64),
             self.eos_token,
             torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
             ],
            dim=0, #concat accross rows
        )
        # add only <s> token
        decoder_input = torch.cat(
            [self.sos_token,
             torch.tensor(dec_input_tokens, dtype=torch.int64),
             torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
             ],
            dim=0,

        )
        # add only </s> token
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),

        ],
            dim=0,
        )
        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # seq_len
            "decoder_input": decoder_input, # seq_len
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() ,# (1,1,seq_len)  (batch dim, head_dim, keys dim) --> .unsqueeze(0).unsqueeze(0) to make it compatible with multiheadattention
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)) , #(1,seq_len) &(1, seq_len)-->(batch, seq_len) & (1,seq_len, seq_len) final shape--> (1,1,seq_len) & (1,seq_len,seq_len)--> (1,seq_len,seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,

        }
def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size), diagonal=1).type(torch.int) # diagonal =1 , upper triangle as 1, .int for conversion, ==0 means positions below diagonals is true
    return mask == 0



# mask1 = torch.tril(torch.ones(1, size, size), diagonal=1).int()
# # Upper triangle (excluding diagonal)
# mask2 = torch.triu(torch.ones(1, size, size), diagonal=1).int()
