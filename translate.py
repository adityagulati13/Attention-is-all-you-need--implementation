from pathlib import Path
from config import get_config, latest_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset
import torch
import sys


def translate(sentence: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)
    config = get_config()
    #getitng pretrained tokenizers
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))

    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config["seq_len"],
                              config['seq_len'], d_model=config['d_model']).to(device)

    #load pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)  # file in which model pretrained is stored
    model.load_state_dict(state["model_state_dict"])

    ##if the sentence is a number --> use it as an index to test set
    label = ""
    if type (sentence) == int or sentence.isdigit():
        id = int(sentence)
        ds = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='all')
        ds = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

        sentence = ds[id]['src_text']
        label = ds[id]["tgt_txt"]
    seq_len = config['seq_len']

    # sentence translation
    model.eval()
    with torch.no_grad():
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device) # encoder ready source input

        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        encoder_output = model.encode(source, source_mask)


        # decoder input
        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)

        #print the source sentence and targent start
        if label != "": print(f"{f'ID: ':>12}{id}")
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "": print(f"{f'TARGET: ':>12}{label}")
        print(f"{f'PREDICTED: ':>12}", end='')
        #word by word translation
        while decoder_input.size(1) < seq_len:
            #masking the targents and then calculate output
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(
                torch.int).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
             #next_token projection

            prob = model.project(out[:,-1])
            _, next_word = torch.max(prob, dim=1)
            #concat
            decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
            #translated word
            print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')
            #if sent--> end break:
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break
    return tokenizer_tgt.decode(decoder_input[0].tolist())


translate(sys.argv[1] if len(sys.argv) > 1 else "I am not a very good a student.")


