import argparse
from contextlib import nullcontext
from dotmap import DotMap
from hydra import compose, initialize
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm
from transformers import set_seed
from warnings import filterwarnings
filterwarnings("ignore")

from src.tokenization import build_tokenizer
from src.data import build_dataset, build_loader
from src.model import build_model_from_scratch, DECODER_BASED
from src.evaluate import get_tokenwise_accuracy, get_instancewise_accuracy

def evaluate(args):    
    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides
    min_n_digits = args.min_n_digits
    max_n_digits = args.max_n_digits
    step = args.step

    # Hydra Compose
    initialize(version_base=None, config_path=config_path) 
    cfg = compose(config_name=config_name, overrides=overrides)

    # Bring model configs
    logging_path = os.path.join("log", cfg.group_name, cfg.exp_name, f"seed{cfg.seed}_seedData{cfg.seed_data}")
    with open(os.path.join(logging_path, "cfg.json")) as f:
        dict_cfg = json.load(f)
    for k in dict_cfg['model']:
        cfg.model[k] = dict_cfg['model'][k]

    # device
    if cfg.device=='cpu':
        device = torch.device('cpu')
        device_type = 'cpu'
    elif str(cfg.device).startswith('cuda:'):
        os.environ["CUDA_VISIBLE_DEVICES"]= cfg.device.split(":")[-1]
        device = torch.device('cuda')
        device_type = 'cuda'

    # Data type
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.cuda.amp.autocast(dtype=ptdtype) if device_type == 'cuda' else nullcontext()
    
    # Training Misc
    model_name = cfg.model.model_name
    logging_path = f"log/{cfg.group_name}/{cfg.exp_name}/seed{cfg.seed}_seedData{cfg.seed_data}"
    print(logging_path)

    # Tokenizer
    if "IndexHints" in cfg.task.train.dataset_cls:
        cfg.task.vocab = " ".join(list(map(str, range(int(cfg.task.max_position)+10))) + \
                                  [cfg.task.symbol, '='])
    tokenizer = build_tokenizer(cfg)
    if "IndexHints" in cfg.task.train.dataset_cls:
        id_index_hint_begin = tokenizer.token_to_id('10')
        id_index_hint_end = tokenizer.token_to_id(str(int(cfg.task.max_position)+9))
    
    # Model
    model = build_model_from_scratch(cfg, tokenizer, device)
    if compile:
        model = torch.compile(model)

    # get pretrained model
    model_path = os.path.join(logging_path, f'last_{model_name}.pt')
    if cfg.get('best', False) or not os.path.exists(model_path):
        print("Testing Best Model")
        mode = 'best'
        model_path = os.path.join(logging_path, f'best_{model_name}.pt')
    if model_path.startswith(logging_path+'/last'):
        print("Testing Last Model")
        mode = 'last'
    if not os.path.exists(os.path.join(model_path)):
        print("No model exists... Returning...")
        return
    # if os.path.exists(os.path.join(logging_path, f'performances_EVAL_{mode}.json')):
    #     print("Evaluation is already done.")
    #     return
    model.load_state_dict(torch.load(model_path, map_location=torch.device(cfg.device)))
    model.eval()

    losses = []
    tokenwise_accuracies = []
    instancewise_accuracies = []

    # step = 1
    for n_digit_A in reversed(range(min_n_digits, max_n_digits+1, step)):
        losses_ = []
        tokenwise_accuracies_ = []
        instancewise_accuracies_ = []
        for n_digit_B in reversed(range(min_n_digits, max_n_digits+1, step)):
            cfg.task.val_long.min_n_digit_A = n_digit_A
            cfg.task.val_long.max_n_digit_A = n_digit_A
            cfg.task.val_long.min_n_digit_B = n_digit_B
            cfg.task.val_long.max_n_digit_B = n_digit_B
            print(f"{n_digit_A}-digit + {n_digit_B}-digit")

            # Random seed
            set_seed(seed=999)

            # Dataset / Dataloader
            dataset = build_dataset(cfg,verbose=False)
            loader = build_loader(cfg, dataset, tokenizer, device)

            phase = 'val_long'
        
            # Training Epoch
            pbar = tqdm(loader[phase])
            loss_sum = 0.
            tokenwise_correct_sum = 0
            num_tokens_sum = 0
            instancewise_correct_sum = 0
            for batch_idx, model_inputs in enumerate(pbar):
                if "IndexHints" in cfg.task.train.dataset_cls and cfg.task.get('hide_index_hints', False):
                    model_inputs['labels'] = torch.where(
                        torch.logical_and(model_inputs['labels'] >= id_index_hint_begin, 
                                        model_inputs['labels'] <= id_index_hint_end),
                        -100,
                        model_inputs['labels']
                    )
                with ctx:
                    model_output = model(**model_inputs)
                    loss = model_output.loss
                with torch.no_grad():
                    batchsize = len(model_inputs['input_ids'])
                    loss_sum += loss.float() * batchsize
                    logits = model_output.logits
                    pred = torch.argmax(logits, dim=-1)
                    tokenwise_correct, num_tokens = get_tokenwise_accuracy(cfg, pred, model_inputs['labels'], tokenizer.pad_token_id, division=False)
                    instancewise_correct, _ = get_instancewise_accuracy(cfg, pred, model_inputs['labels'], tokenizer.pad_token_id, division=False)
                    tokenwise_correct_sum += tokenwise_correct
                    num_tokens_sum += num_tokens
                    instancewise_correct_sum += instancewise_correct
                    pbar.set_description(f"Loss:{loss:.3g} | TokenAcc:{tokenwise_correct/num_tokens:.3g} | InstAcc:{instancewise_correct/batchsize:.3g}") 
            
            # Logging
            loss_avg = loss_sum.item()/len(dataset[phase])
            tokenwise_accuracy_avg = (tokenwise_correct_sum/num_tokens_sum).item()
            instancewise_accuracy_avg = instancewise_correct_sum.item()/len(dataset[phase])
            print(f"seed({cfg.seed},{cfg.seed_data}) (A,B)=({n_digit_A},{n_digit_B}) Loss {loss_avg:.6f} TokenAcc {tokenwise_accuracy_avg:.6f} InstAcc {instancewise_accuracy_avg:.6f}")
            losses_.append(loss_avg)
            tokenwise_accuracies_.append(tokenwise_accuracy_avg)
            instancewise_accuracies_.append(instancewise_accuracy_avg)
        
        losses.append(losses_[::-1])
        tokenwise_accuracies.append(tokenwise_accuracies_[::-1])
        instancewise_accuracies.append(instancewise_accuracies_[::-1])
    
    # Save loggings
    X = np.arange(min_n_digits, max_n_digits+1, step)
    perf_dict = {
        'X': X.tolist(),
        'losses': losses[::-1],
        'tokenwise_accuracies': tokenwise_accuracies[::-1],
        'instancewise_accuracies': instancewise_accuracies[::-1]
    }
    with open(os.path.join(logging_path, f'performances_EVAL_{mode}.json'), 'w') as f:
        json.dump(perf_dict, f, indent=2)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_path', type=str,  default='./configs')
    parser.add_argument('--config_name', type=str,  default='config')
    parser.add_argument('--min_n_digits',type=int,  default=1)
    parser.add_argument('--max_n_digits',type=int,  default=100)
    parser.add_argument('--step',        type=int,  default=1)
    parser.add_argument('--overrides',   type=str,  default=[],  nargs='*')
    args = parser.parse_args()

    evaluate(vars(args))