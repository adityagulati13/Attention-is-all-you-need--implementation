from pathlib import Path
# def get_config():
#     return {
#         "batch_size": 8,
#         "num_epochs": 20,
#         "lr": 10**-4,
#         "seq_len": 350,#max len of input and output seq
#         "d_model": 512, # dimensionality of model embedding
#         "data_source": "Helsinki-NLP/opus_books",#HF datasource
#         "lang_src": "en", "lang_tgt": "it", #src and tgt lang
#         "model_folder": "weights", #Folder to store modek checkpoints
#         "model_basename": "tmodel_", # prefix for save mmodel file
#         "preload" : "latest", # to reusume from latest checkpoint
#         "tokenizer_file": "tokenizer_{0}.json", # to store tokenizer where {0} to be replaced by lang
#         "experiment_name": "runs/tmodel" # to store Tensorboard logs
#     }
def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 3,
        "lr": 10**-4,
        "seq_len": 128,#max len of input and output seq
        "d_model": 512, # dimensionality of model embedding
        "data_source": "Helsinki-NLP/opus_books",#HF datasource
        "lang_src": "en", "lang_tgt": "it", #src and tgt lang
        "model_folder": "weights", #Folder to store modek checkpoints
        "model_basename": "tmodel_", # prefix for save mmodel file
        "preload" : "latest", # to reusume from latest checkpoint
        "tokenizer_file": "tokenizer_{0}.json", # to store tokenizer where {0} to be replaced by lang
        "experiment_name": "runs/tmodel" # to store Tensorboard logs
    }
def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['data_source']}_{config['model_folder']}"  #folder to store model weights
    model_filename = f"{config['model_basename']}{epoch}.pt"  #joining model with epoch
    return str(Path('.') / model_folder / model_filename) # returns full path of model wts
#findning latest files
def latest_weights_file_path(config):
    model_folder = f"{config['data_source']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])  # getting latest weights

