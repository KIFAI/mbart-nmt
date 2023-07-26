import os
import json
import random
import torch
import evaluate
import numpy as np

def seed_worker(worker_id):
    """
    Helper function for reproducible behavior to set the seed in multi-process dataloader
    Use worker_init_fn() and generator to preserve reproducibility.
    refer to https://pytorch.org/docs/stable/notes/randomness.html
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(random_seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).
    Args:
        seed (`int`): The seed to set.
    """
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def log_args(args, logger):
    '''
    Logging from using transmers.logging
    params:
        args : argparse.Namespace
        logger : transformers.logging.get_logger()
    '''
    args_dict = vars(args)
    max_len = max([len(k) for k in args_dict.keys()])
    fmt_string = "\t%" + str(max_len) + "s : %s"
    logger.info("Arguments:")
    for key, value in args_dict.items():
        logger.info(fmt_string, key, value)

def load_json(f_path):
    '''
    Load json data, like model configs
    params:
            f_path : json file path
    '''
    with open(f_path, 'r') as f:
        data = json.load(f)

    return data

def get_loss_fn(args):
    '''
    -100 will be automatically ignored by PyTorch loss function, which is Cross Entropy
    1) For MLM tasks that only predict what the [mask] token is, ignore anything other than the [mask] token.
    2) Ignore BOS/SOS/Padding Tokens for denoising tasks that require not only predicting what the [mask] token is, but also restoring the origin
    params:
        args : Training args with label smoothing of cross entropy loss function hyper params
    '''
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing)

    return loss_fn

def postprocess_text(preds, labels, tokenizer):
    '''
    args:
        preds : generated tokens
        labels : labels
        tokenizer : your tokenizer
    '''
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    return decoded_preds, decoded_labels

def compute_loss_and_logits(model, batch_inputs, labels, args):
    '''
    Compute loss and logits from model and loss function
    params:
            model: MBartForConditionalGeneration Class Module
            batch_inputs: Input ids, Attention masks, Decoder inputs
            labels: Labels with -100 tokens ids. tokens with -100 id could be different by pretraining task
            args: Training argument, a boolean value containing whether 'ignore_tokens_ixs_for_loss' is true
    '''
    loss_fn = get_loss_fn(args)
    if args.ignore_tokens_ixs_for_loss:
        # force training to ignore token ixs, which is -100
        logits = model(**batch_inputs)[0]
        if torch.distributed.is_initialized():
            loss = loss_fn(logits.view(-1, model.module.config.vocab_size), labels.view(-1))
        else:
            loss = loss_fn(logits.view(-1, model.config.vocab_size), labels.view(-1))
    else:
        # compute usual loss via models
        loss, logits = model(**batch_inputs, labels=labels)[:2]

    return loss, logits

def load_metric_modules(args, metric_types:list=['accuracy', 'recall', 'precision', 'f1', 'sacrebleu']):
    '''
    Load metric modules from huggingface evaluate library
    params:
        metric_types : List of metric types. ex) ['accuracy', 'recall', 'precision', 'f1']
        args : argparse.NameSpace
    '''
    metric_modules = {}

    for m_t in metric_types:
        if m_t.find(args.early_stop_metric) != -1:
            metric_modules[m_t] = evaluate.load(m_t, keep_in_memory=False)

    return metric_modules

def compute_metrics(metrics:list):
    '''
    Compute batch predictions and references
    params:
        metrics : List of EvaluationModules. ex) Accuracy, Recall, Precision, F1, etc..
    '''
    scores = {}

    for metric in metrics:
        if metric.name in ["accuracy", "sacrebleu"]:
            scores.update(metric.compute())
        elif metric.name == "f1":
            scores.update(metric.compute(average='weighted'))
        else:
            #recall, precision
            scores.update(metric.compute(average='weighted', zero_division=0))

    return scores

def save_model(accelerator, args, model, tokenizer):
    '''
    Save model in specific path
        Args:
            accelerator : `accelerator` object
            args : argparse.Namespace
            model : pytorch model
            tokenizer : tokenizer
    '''
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(os.path.join(args.output_dir, args.exp_name), save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(os.path.join(args.output_dir, args.exp_name))
