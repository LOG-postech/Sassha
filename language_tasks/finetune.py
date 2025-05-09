# Acknowledgement: this code is based on : https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py

import sys
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import torch
from transformers import set_seed

from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
import torch.nn as nn

from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import wandb
os.environ["WANDB__SERVICE_WAIT"] = "300"

# Load optimizers
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('optimizers'))))
from optimizers import get_optimizer

# load hessain power scheduler
from optimizers.hessian_scheduler import ConstantScheduler, ProportionScheduler, LinearScheduler, CosineScheduler


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.41.0.dev0")

#logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    parser.add_argument(
        "--clip_norm",
        type=int,
        default=10,
        help="Clip norm for SGD optimizer (default: %(default)s).",
    )

    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    
    parser.add_argument('--optimizer', type=str, default='sassha', help='choose optim')

    # Second-order optimization settings
    parser.add_argument("--n_samples", default=1, type=int, help="the number of sampling")
    parser.add_argument('--betas', type=float, nargs='*', default=[0.9, 0.999], help='betas')
    parser.add_argument("--eps", default=1e-4, type=float, help="add a small number for stability")
    parser.add_argument("--lazy_hessian", default=10, type=int, help="Delayed hessian update")
    parser.add_argument("--clip_threshold", default=0.01, type=float, help="sophia clipping")
    
    # Hessian power scheduler
    parser.add_argument('--hessian_power_scheduler', type=str, default='constant', help="choose Hessian power 1. 'constant', 2. 'proportion', 3. 'linear', 4. 'cosine'")
    parser.add_argument('--max_hessian_power', type=float, default=1)
    parser.add_argument('--min_hessian_power', type=float, default=0.5)
    parser.add_argument('--min_lr', type=float, default=0.0, help="the minimum value of learning rate")

    # Sharpness minimization settings
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for sharpness minimization")
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive sharpness")
    parser.add_argument('--project_name', type=str, default='project_name', help="project_name")
    
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()
    
    # Control randomness
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)    
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load Dataset
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("nyu-mll/glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer

    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    
    # Get config
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        trust_remote_code=args.trust_remote_code,
    )

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id

    # Get model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        trust_remote_code=args.trust_remote_code,
    )

    # 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}

        else:
            pass
            
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    # Set target padding
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    wandb_name = f"{args.task_name}-{args.optimizer}-{args.lr}-{args.weight_decay}-{args.eps}"
    
    # get an optimizer
    optimizer, create_graph, two_steps = get_optimizer(model, args)    

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # select a hessian power scheduler
    if args.optimizer == 'sassha':
        if args.hessian_power_scheduler == 'constant':
            hessian_power_scheduler = ConstantScheduler(
                T_max=args.num_train_epochs*len(train_dataloader), 
                max_value=0.5,
                min_value=0.5)
        
        elif args.hessian_power_scheduler == 'proportion':
            hessian_power_scheduler = ProportionScheduler(
                pytorch_lr_scheduler=scheduler,
                max_lr=args.lr,
                min_lr=args.min_lr,
                max_value=args.max_hessian_power,
                min_value=args.min_hessian_power)
        
        elif args.hessian_power_scheduler == 'linear':
            hessian_power_scheduler = LinearScheduler(
                T_max=args.num_train_epochs*len(train_dataloader), 
                max_value=args.max_hessian_power,
                min_value=args.min_hessian_power)
        
        elif args.hessian_power_scheduler == 'cosine':
            hessian_power_scheduler = CosineScheduler(
                T_max=args.num_train_epochs*len(train_dataloader), 
                max_value=args.max_hessian_power,
                min_value=args.min_hessian_power)
        
        optimizer.hessian_power_scheduler = hessian_power_scheduler

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Get the metric function
    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name, experiment_id=wandb_name)
    else:
        metric = evaluate.load("accuracy", experiment_id=wandb_name)
    
    def compute_metric(eval_pred):
        predictions, labels = eval_pred
        if args.task_name != "stsb":
            predictions = torch.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # wandb
    wandb_project = args.project_name
    wandb.init(project=wandb_project, name=wandb_name)
    wandb.config.update(args)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        
        for step, batch in enumerate(active_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            if two_steps:
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                total_loss += loss.item()
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if args.optimizer == 'sassha':
                    optimizer.perturb_weights(zero_grad=True)

                elif args.optimizer in ['samsgd', 'samadamw']:
                    optimizer.first_step(zero_grad=True)
                
                model(**batch).loss.backward(create_graph=create_graph)
                
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    if args.optimizer == 'sassha':
                        optimizer.unperturb()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    elif args.optimizer in ['samsgd', 'samadamw']:
                        if args.grad_clip_norm != 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                        optimizer.second_step(zero_grad=True)
                    
                    lr_scheduler.step()
                    progress_bar.update(1)
                    completed_steps += 1

            else:
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                total_loss += loss.item()
                loss = loss / args.gradient_accumulation_steps
                loss.backward(create_graph=create_graph)

                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    
                    if args.grad_clip_norm != 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                    progress_bar.update(1)
                    completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        all_labels, all_logits = [], []
        val_loss = 0
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                all_labels.append(batch['labels'])
                all_logits.append(outputs.logits)

        all_labels = torch.cat(all_labels, dim=0)
        all_logits = torch.cat(all_logits, dim=0)
        eval_metric = compute_metric((all_logits, all_labels))

        wandb.log({
            "accuracy" if args.task_name is not None else "glue": eval_metric,
            "train_loss": total_loss / len(train_dataloader),
            "val_loss": val_loss / len(eval_dataloader),
            "epoch": epoch,
            "step": completed_steps,
            "lr": optimizer.param_groups[0]['lr'],
            "hessian_power": optimizer.hessian_power_t if args.optimizer == 'sassha' else 0,
            })

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )

        model.eval()
        all_labels, all_logits = [], []
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                all_labels.append(batch['labels'])
                all_logits.append(outputs.logits)

        all_labels = torch.cat(all_labels, dim=0)
        all_logits = torch.cat(all_logits, dim=0)
        eval_metric = compute_metric((all_logits, all_labels))

        wandb.log({
            "accuracy-mm" if args.task_name is not None else "glue": eval_metric,
            })

    if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f)


if __name__ == "__main__":
    main()
