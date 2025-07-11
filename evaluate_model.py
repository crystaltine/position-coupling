import argparse
from contextlib import nullcontext
from dotmap import DotMap
from hydra import compose, initialize
import json
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import os
import torch
from tqdm import tqdm
from transformers import set_seed
from warnings import filterwarnings
filterwarnings("ignore")

from src.tokenization import build_tokenizer
from src.data import build_dataset, build_loader
from src.model import build_model_from_scratch, DECODER_BASED
from src.evaluate import get_tokenwise_accuracy, get_instancewise_accuracy, get_parity_accuracy, get_answerwise_accuracy

def evaluate(args):    
    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides
    min_n_digits = args.min_n_digits
    max_n_digits = args.max_n_digits
    eval_step = args.step_digits
    compile = args.compile

    # Hydra Compose
    initialize(version_base=None, config_path=config_path) 
    cfg = compose(config_name=config_name, overrides=overrides)

    # Bring model configs
    logging_path = os.path.join("log", cfg.group_name, cfg.exp_name, f"seed{cfg.seed}_seedData{cfg.seed_data}")
    with open(os.path.join(logging_path, "cfg.json")) as f:
        dict_cfg = json.load(f)
    for k in dict_cfg['model']:
        cfg.model[k] = dict_cfg['model'][k]
        # print(k, dict_cfg['model'][k], eval(f"cfg.model.{k}"))

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
    # print(logging_path)

    # Tokenizer
    if "IndexHints" in cfg.task.train.dataset_cls:
        cfg.task.vocab = " ".join(list(map(str, range(int(cfg.task.max_position)+10))) + \
                                  [cfg.task.symbol, '='])
    tokenizer = build_tokenizer(cfg)
    if "IndexHints" in cfg.task.train.dataset_cls:
        id_index_hint_begin = tokenizer.token_to_id('10')
        id_index_hint_end = tokenizer.token_to_id(str(int(cfg.task.max_position)+9))
    id_0 = tokenizer.token_to_id('0')
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    eq_token_id = tokenizer.token_to_id('=')
    sep_token_id = tokenizer.token_to_id('>') if ('>' in tokenizer.get_vocab()) else eq_token_id
    
    # Model
    model = build_model_from_scratch(cfg, tokenizer, device)
    if compile:
        model = torch.compile(model)

    # get pretrained model
    model_path = os.path.join(logging_path, f'last_{model_name}.pt')
    if cfg.get('best', False):
        print("Testing Best Model:", logging_path)
        mode = 'best'
        model_path = os.path.join(logging_path, f'best_{model_name}.pt')
    if model_path.startswith(logging_path+'/last'):
        print("Testing Last Model:", logging_path)
        mode = 'last'
    if not os.path.exists(model_path):
        print("No model exists... Returning...:", logging_path)
        return
    # if os.path.exists(os.path.join(logging_path, f'performances_EVAL_{mode}.json')):
    #     print("Evaluation is already done.:", logging_path)
    #     return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    d_positions = getattr(cfg.model, 'd_positions', None)
    is_parity = cfg.task.train.dataset_cls.startswith("Parity")

    cfg.task.train.min_n_digits=1
    cfg.task.train.max_n_digits=1
    cfg.task.train.n_data=1
    cfg.task.val.min_n_digits=1
    cfg.task.val.max_n_digits=1
    cfg.task.val.n_data=1

    losses = []
    tokenwise_accuracies = []
    instancewise_accuracies = []
    answerwise_accuracies = []
    if is_parity:
        parity_accuracies = []

    for n_digits in reversed(range(min_n_digits, max_n_digits+1, eval_step)):
        try:
            cfg.task.val_long.min_n_digits = n_digits
            cfg.task.val_long.max_n_digits = n_digits
        except omegaconf.errors.ConfigAttributeError:
            cfg.task.val_long.min_n_len = n_digits
            cfg.task.val_long.max_n_len = n_digits
        print(f"N={n_digits}")

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
        answerwise_correct_sum = 0
        if is_parity:
            parity_correct_sum = 0
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
                loss = model_output.loss.float().item()
            with torch.no_grad():
                batchsize = len(model_inputs['input_ids'])
                loss_sum += loss * batchsize
                logits = model_output.logits
                pred = torch.argmax(logits, dim=-1)
                tokenwise_correct, num_tokens = get_tokenwise_accuracy(cfg, pred, model_inputs['labels'], pad_token_id, division=False)
                instancewise_correct, _ = get_instancewise_accuracy(cfg, pred, model_inputs['labels'], pad_token_id, division=False)
                answerwise_correct, _ = get_answerwise_accuracy(cfg, pred, model_inputs['labels'], eos_token_id=eos_token_id, sep_token_id=sep_token_id, division=False)
                tokenwise_correct_sum += tokenwise_correct.item()
                num_tokens_sum += num_tokens.item()
                instancewise_correct_sum += instancewise_correct.item()
                answerwise_correct_sum += answerwise_correct.item()
                if is_parity:
                    parity_correct, _ = get_parity_accuracy(cfg, pred, model_inputs['labels'], eos_token_id, division=False, return_arr=False)
                    parity_correct_sum += parity_correct.item()                    
                pbar.set_description(f"Loss:{loss:.3g}"
                                     f" | TokenAcc:{tokenwise_correct/num_tokens:.3g}"
                                     f" | InstAcc:{instancewise_correct/batchsize:.3g}"
                                     f" | AnsAcc:{answerwise_correct/batchsize:.3g}" \
                                     + (f" | ParityAcc:{parity_correct/batchsize:.3g}" if is_parity else "")) 
        
        # Logging
        loss_avg = loss_sum/len(dataset[phase])
        tokenwise_accuracy_avg = (tokenwise_correct_sum/num_tokens_sum)
        instancewise_accuracy_avg = instancewise_correct_sum/len(dataset[phase])
        answerwise_accuracy_avg = answerwise_correct_sum/len(dataset[phase])
        if is_parity:
            parity_accuracy_avg = parity_correct_sum/len(dataset[phase])
        print(f"seed({cfg.seed},{cfg.seed_data}) N={n_digits}"
              f" Loss {loss_avg:.6f}"
              f" TokenAcc {tokenwise_accuracy_avg:.6f}"
              f" InstAcc {instancewise_accuracy_avg:.6f}"
              f" AnsAcc {answerwise_accuracy_avg:.6f}" \
              + (f" ParityAcc:{parity_accuracy_avg:.3g}\n" if is_parity else "\n"))
        losses.append(loss_avg)
        tokenwise_accuracies.append(tokenwise_accuracy_avg)
        instancewise_accuracies.append(instancewise_accuracy_avg)
        answerwise_accuracies.append(answerwise_accuracy_avg)
        if is_parity:
            parity_accuracies.append(parity_accuracy_avg)
    
    # Save loggings
    X = np.arange(min_n_digits, max_n_digits+1, eval_step)
    perf_dict = {
        'X': X.tolist(),
        'losses': losses[::-1],
        'tokenwise_accuracies': tokenwise_accuracies[::-1],
        'instancewise_accuracies': instancewise_accuracies[::-1],
        'answerwise_accuracies': answerwise_accuracies[::-1]
    }
    if is_parity:
        perf_dict['parity_accuracies'] = parity_accuracies[::-1]
    with open(os.path.join(logging_path, f'performances_EVAL_{mode}.json'), 'w') as f:
        json.dump(perf_dict, f, indent=2)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_path', type=str,  default='./configs')
    parser.add_argument('--config_name', type=str,  default='config')
    parser.add_argument('--min_n_digits',type=int,  default=1)
    parser.add_argument('--max_n_digits',type=int,  default=100)
    parser.add_argument('--step_digits', type=int,  default=1)
    parser.add_argument('--compile',   action='store_true')
    parser.add_argument('--overrides',   type=str,  default=[],  nargs='*')
    args = parser.parse_args()

    evaluate(vars(args))