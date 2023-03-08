import os
import gc
import logging
import math
import argparse
import datasets
import torch
import transformers

from utils import *

from datetime import timedelta
from tqdm.auto import tqdm

#from data_loader.prepare_data import PreProcessor, Preparator
#from data_loader.data_collator import DataCollatorForDenoisingTasks
from data_loader.nmt_loader import NmtDataLoader, Processor

from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType, DeepSpeedPlugin
from accelerate.utils import InitProcessGroupKwargs, DummyOptim, DummyScheduler

from transformers.optimization import AdamW
from transformers.models.mbart.configuration_mbart import MBartConfig
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, get_scheduler

logger = transformers.logging.get_logger()
transformers.logging.set_verbosity_info()
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()

def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        default="/home/jihyo/translation/repo/mbart-nmt",
        type=str,
    )
    parser.add_argument(
        "--ds_config_path",
        type=str
    )
    parser.add_argument(
        "--tokenizer_path",
        default="./src/plm/reduced_hf_mbart50_m2m",
        type=str
    )
    parser.add_argument(
        "--corpus_path",
        default="./src/train_corpus/cased_corpus_exp",
        type=str,
    )
    parser.add_argument(
        "--plm_path",
        default="./src/plm/reduced_hf_mbart50_m2m_v2",
        type=str,
    )
    parser.add_argument(
        "--num_proc",
        default=8,
        type=int
    )
    parser.add_argument(
        "--max_token_length",
        default=512,
        type=int
    )
    parser.add_argument(
        "--batch_size",
        default=18,
        type=int
    )
    parser.add_argument(
        "--mixed_precision",
        default="fp16",
        type=str,
        help="Whether or not to use mixed precision training (fp16 or bfloat16). Choose from ‘no’,‘fp16’,‘bf16’."
    )
    parser.add_argument(
        "--output_dir",
        default=f"{os.path.join(os.getcwd(), 'src/ftm')}",
        type=str
    )
    parser.add_argument(
        "--exp_name",
        default='mbart02_enko',
        type=str,
    )
    parser.add_argument(
        "--src_lang",
        default="en_XX",
        type=str
    )
    parser.add_argument(
        "--tgt_lang",
        default="ko_KR",
        type=str,
    )
    parser.add_argument(
        '--drop_case',
        action='store_true',
        help='Whether to drop sequence if it is longer than max_token length',
    )
    parser.add_argument(
        '--bi_direction',
        action='store_true',
        help='Whether to train bi-direcional NMT Engine instead of uni_directional training.',
    )
    parser.add_argument(
        '--packing_data',
        action='store_true',
        help='Merge sentences into segments',
    )
    parser.add_argument(
        "--packing_size",
        default=256,
        type=int
    )
    parser.add_argument(
        '--hybrid',
        action='store_true',
        help='Prepare train data using sents & segments unit',
    )
    parser.add_argument(
        '--adam_beta1',
        default=0.9,
        type=float
    )
    parser.add_argument(
        '--adam_beta2',
        default=0.999,
        type=float
    )
    parser.add_argument(
        '--adam_epsilon',
        default=1e-08,
        type=float
    )
    parser.add_argument(
        '--learning_rate',
        default=5e-5,
        type=float
    )
    parser.add_argument(
        '--weight_decay',
        default=0.0,
        type=float
    )
    parser.add_argument(
        '--lr_scheduler_type',
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        '--num_warmup_steps',
        default=0,
        type=int
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=4,
        type=int,
        help='When the batch size is small, it can be used if you want to have the same effect as the batch size as many as a parameter multiple.'
    )
    parser.add_argument(
        "--num_epochs",
        default=2,
        type=int
    )
    parser.add_argument(
        "--eval_check_interval",
        default=0.25,
        type=float,
        help="Check validation set X times during a training epoch"
    )
    parser.add_argument(
        '--label_smoothing',
        default=0.2,
        type=float
    )
    parser.add_argument(
        '--pad_token_id',
        default=1,
        type=int
    )
    parser.add_argument(
        '--ignore_tokens_ixs_for_loss',
        action='store_true'
    )
    parser.add_argument(
        '--trainer_seed',
        default=42,
        type=int
    )
    parser.add_argument(
        '--early_stop_metric',
        default='f1',
        type=str
    )
    parser.add_argument(
        '--patience',
        default=5,
        type=int
    )
    args = parser.parse_args()

    return args

def b2mb(x):
    '''
    Converting Bytes to Megabytes
    '''
    return int(x / 2**20)

class TorchTraceMemAlloc:
    '''
    This context manager is used to track the peak memory usage of the process
    '''
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        return self

    def __exit__(self, *exc):
        '''
        delta refers to the change in the amount of memory
        '''
        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)
        print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")

def get_dataloaders(accelerator: Accelerator, model, tokenizer, args):
    """
    Creates a set of `DataLoader`s for the custom dataset,
    using "mBart50TokenizerFast" as the tokenizer.
    Args:
        accelerator ('Accelerator'):
            'Accelerator' object
        tokenizer :
            'mBart50Tokenizer' object
        args ('ArgumentParser'):
            'ArgumentParser' object
    """
    with accelerator.main_process_first():
        preprocessor = Processor(tokenizer, args.src_lang, args.tgt_lang, args.max_token_length, args.drop_case, args.bi_direction)
        preparator = NmtDataLoader(tokenizer, preprocessor, args.corpus_path, args.packing_data, args.packing_size, args.hybrid)
        segment_datasets = preparator.get_tokenized_dataset(batch_size=20000, num_proc=args.num_proc)
        #preprocessor = PreProcessor(tokenizer, sent_tokenizer)
        #preparator = Preparator(preprocessor, args)
        #segment_datasets = preparator.prepare_batched_segments(batch_size=20000)

    #Denoising Collator sets each seed with a number for each device, and inputs that are noised by text infilling / sentence permutation are dynamically created.
    #data_collator = DataCollatorForDenoisingTasks(tokenizer=tokenizer, max_token_length=args.max_token_length, seed=torch.cuda.current_device())
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding='max_length', max_length=512)

    g = torch.Generator()
    g.manual_seed(args.trainer_seed)

    train_dataloader = DataLoader(segment_datasets['train'].with_format("torch"), batch_size=args.batch_size, collate_fn=data_collator, shuffle=True,
            num_workers=torch.cuda.device_count()*2, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    eval_dataloader = DataLoader(segment_datasets['valid'].with_format("torch"), batch_size=args.batch_size, collate_fn=data_collator, shuffle=True,
            num_workers=torch.cuda.device_count()*2, worker_init_fn=seed_worker, generator=g, pin_memory=True)

    return train_dataloader, eval_dataloader


def training_functions(args):
    # Initialize accelerator
    ipg_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))

    if args.ds_config_path is not None:
        with open(args.ds_config_path, "r") as f:
            ds_config = json.load(f)
        if args.mixed_precision == 'bf16' and (args.mixed_precision not in ds_config.keys()):
            try:
                del ds_config['fp16']
            except Exception as ex:
                print(ex)
            ds_config.update({args.mixed_precision: {'enabled': True}})
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)
        if int(os.environ.get('LOCAL_RANK', -1)) == 0:
            logger.info(f"Using pre-defined deep speed configuration about optimizer, scheduler, etc.. Refer to below aceelerator's state\n")
    else:
        deepspeed_plugin = None
        if int(os.environ.get('LOCAL_RANK', -1)) == 0:
            logger.info(f"Using user defined arguments about optimizer, scheduler, etc...\n")
    
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps, cpu=False, deepspeed_plugin=deepspeed_plugin,
                            log_with="tensorboard", logging_dir=os.path.join(args.output_dir, args.exp_name), kwargs_handlers=[ipg_handler])
    accelerator.free_memory()
    logger.info("\n" + repr(accelerator.state) + "\n")
    
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO
                )
        datasets.utils.logging.set_verbosity_warning()
        transformers.logging.set_verbosity_info()

        # NOTE Remove redundant logs
        transformers.logging.get_logger("transformers.configuration_utils").setLevel(logging.WARN)
        transformers.logging.get_logger("transformers.modeling_utils").setLevel(logging.ERROR)
        transformers.logging.get_logger("transformers.tokenization_utils_base").setLevel(logging.WARN)
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.logging.set_verbosity_error()
    
    if accelerator.is_main_process:
        #logging for user defined arguments
        log_args(args, logger)

    # Helpful function for reproducible training from setting the seed in random, numpy, torch, and/or tf
    set_seed(args.trainer_seed)
    
    tokenizer = MBart50TokenizerFast.from_pretrained(os.path.join(args.base_path, args.tokenizer_path))
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = MBartForConditionalGeneration.from_pretrained(args.plm_path)

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    accelerator.wait_for_everyone()
    with accelerator.main_process_first():
        train_dataloader, eval_dataloader = get_dataloaders(accelerator, model, tokenizer, args)
        
    config = MBartConfig(args.plm_path)

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        logger.info(f"\n{config}\n")

    '''
    If you have pre-trained a MBART model using the MBartForConditionalGeneration class and you then load the pre-trained weights from using the MBartForSequenceClassification.from_pretrained() method to finetune on down-stream task, the method will not use weight of the lm_head pretrained from the MBartForConditionalGeneration class. Instead, it will instantiate a new MBartForSequenceClassification model with a sequence classification head (MBartClassificationHead) that is tailored for making predictions on input sequences, and load the pre-trained weights into the new model.
    The from_pretrained() method in the transformers library is designed to work with a variety of different model architectures and it figures out which weights should be loaded where based on the model architecture you specify when you call the method.
    Some weights of the model checkpoint at facebook/mbart-large-50 were not used when initializing MBartForSequenceClassification: ['final_logits_bias', 'lm_head.weight']
    This IS expected if you are initializing MBartForSequenceClassification from the checkpoint of a model trained on another task or with another architecture.
    Some weights of MBartForSequenceClassification were not initialized from the model checkpoint at facebook/mbart-large-50 and are newly initialized: ['classification_head.dense.weight', 'classification_head.out_proj.bias', 'classification_head.out_proj.weight', 'classification_head.dense.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    '''

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
            "weight_decay": 0.0,
        },
    ]
    
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.learning_rate)

    # Get gradient accumulation steps from deepspeed config if available
    if accelerator.state.deepspeed_plugin is not None:
        args.gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"
        ]

    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=math.ceil(args.num_epochs * len(train_dataloader) / args.gradient_accumulation_steps)
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer,
            warmup_num_steps=args.num_warmup_steps,
            total_num_steps=math.ceil(args.num_epochs * len(train_dataloader) / args.gradient_accumulation_steps)
        )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the prepare method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = len(train_dataloader) / args.gradient_accumulation_steps
    total_training_steps = math.ceil(args.num_epochs * num_update_steps_per_epoch)
    eval_interval = math.ceil(len(train_dataloader) * args.eval_check_interval)

    train_progress_bar = tqdm(
                range(total_training_steps),
                disable=not accelerator.is_local_main_process,
                leave=True,
                dynamic_ncols=True,
                desc="Training",
                smoothing=0,
            )
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    accelerator.init_trackers("trainer_logs", experiment_config)

    # Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {total_training_steps}")

    completed_steps = 0
    is_first_ckpt_saved = False
    best_score = 0.0
    cur_patience = 0

    metric_modules = load_metric_modules(args, metric_types=[os.path.join(args.base_path, 'metrics/sacrebleu/sacrebleu.py')])


    # Now we train the model
    for epoch in range(args.num_epochs):
        with TorchTraceMemAlloc() as tracemalloc:
            model.train()
            avg_loss = 0
            for train_step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    labels = batch.pop('labels')
                    loss, _ = compute_loss_and_logits(model, batch, labels, args)
                    avg_loss += loss.item()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                if (train_step + 1) % accelerator.gradient_accumulation_steps == 0 or train_step == len(train_dataloader) - 1:
                    if accelerator.is_main_process:
                        train_progress_bar.update(1)
                        completed_steps += 1
                        accelerator.log({"train_loss": (avg_loss / accelerator.gradient_accumulation_steps)},step=completed_steps)
                        avg_loss = 0
                
                # NOTE Eval on every `val_check_interval`
                if (train_step + 1) % eval_interval == 0 or train_step == len(train_dataloader) - 1:
                    valid_progress_bar = tqdm(
                        range(len(eval_dataloader)),
                        disable=not accelerator.is_local_main_process,
                        dynamic_ncols=True,
                        leave=False,
                        desc="Validation",
                    )
                    with TorchTraceMemAlloc() as tracemalloc:
                        model.eval()
                        samples_seen = 0
                        for valid_step, batch in enumerate(eval_dataloader):
                            generated_tokens_list = []
                            for lang in ["ko_KR", "en_XX"]:
                                filtered_idxs = np.where(batch["input_ids"].cpu().numpy()[:,0]==tokenizer.lang_code_to_id[lang])
                                if batch["input_ids"][filtered_idxs].size()[0] != 0:
                                    with torch.no_grad():
                                        preds = accelerator.unwrap_model(model).generate(batch["input_ids"][filtered_idxs], 
                                                                                       attention_mask=batch["attention_mask"][filtered_idxs])
                                        generated_tokens_list.append(torch.nn.functional.pad(preds, pad=(0, batch["input_ids"].size()[-1]-preds.shape[-1], 0, 0), mode='constant', value=tokenizer.pad_token_id))
                            
                            generated_tokens = torch.cat(generated_tokens_list, dim=0)
                            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                            labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=-100)
                            labels = accelerator.gather(labels).cpu().numpy()
                            
                            decoded_preds, decoded_labels = postprocess_text(generated_tokens, labels, tokenizer)
                            
                            # First we check if it's a distributed system
                            if accelerator.use_distributed:
                                # Then see if we're on the last batch of our eval dataloader
                                if valid_step == len(eval_dataloader) - 1:
                                    # Last batch needs to be truncated on distributed systems as it contains additional samples
                                    decoded_preds = decoded_preds[: len(eval_dataloader.dataset) - samples_seen]
                                    decoded_labels = decoded_labels[: len(eval_dataloader.dataset) - samples_seen]
                                else:
                                    # Otherwise we add the number of samples seen
                                    samples_seen += len(decoded_labels)

                            for m_type, m_module in metric_modules.items():
                                metric_modules[m_type].add_batch(
                                                        predictions=decoded_preds,
                                                        references=decoded_labels)
                                # Empty GPU cache
                                torch.cuda.empty_cache()
                            valid_progress_bar.update(1)

                        scores = compute_metrics(metric_modules.values())
                        try:
                            cur_score = scores[args.early_stop_metric]
                        except :
                            cur_score = scores['score']
                        
                        accelerator.log(scores, step=completed_steps)
                        scores.update({"epoch": epoch, "step": completed_steps})
                        logger.info(f"scores : {scores}")

                        accelerator.wait_for_everyone()
                
                        if not is_first_ckpt_saved:
                            logger.info("Force to save the first checkpoint (Preventing 0.0 score issue)")
                            save_model(accelerator, args, model, tokenizer)
                            is_first_ckpt_saved = True

                        # NOTE Early Stopping
                        if best_score < cur_score:
                            best_score = cur_score
                            cur_patience = 0
                            logger.info(
                                f"valid/{args.early_stop_metric} reached {cur_score*100:.2f}%, saving model to {os.path.join(args.output_dir, args.exp_name)}"
                            )
                            # Save best model
                            save_model(accelerator, args, model, tokenizer)
                        else:
                            cur_patience += 1
                            logger.info(
                                f"valid/{args.early_stop_metric} was not in top 1 (best score: {best_score*100:.2f}% / cur patience: {cur_patience})"
                            )

                if cur_patience >= args.patience:
                    # Stop training
                    logger.info(f"Reached all patience {cur_patience}. So stop training!")
                    break

if __name__ == '__main__':
    args = define_argparser()
    training_functions(args)
