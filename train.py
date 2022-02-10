import argparse
import logging
import os

from datasets import load_dataset, load_metric
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
import torch

from load_data import STSDataLoader
from trainer import Trainer
import utils


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def build_adamw_optimizer(lr, model, num_train_steps, num_warmup_steps, global_step=0):
    last_epoch = -1 if global_step == 0 else global_step
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    for group in optimizer.param_groups:
        group['initial_lr'] = lr
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_steps,
                                                last_epoch=last_epoch)
    return optimizer, scheduler


def train(args):
    # Set logger & seed
    set_seed(args.seed)
    utils.make_dirs(args.model_dir)
    logging.Formatter.converter = utils.kst
    logging.basicConfig(filename=os.path.join(args.model_dir, 'logs_train.txt'),
                        filemode='w', format='%(asctime)s -  %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(args)

    # Set device
    num_gpus = torch.cuda.device_count()
    use_cuda = num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info(f"***** using {device} *****")
    logger.info(f"***** num GPU: {num_gpus} *****")

    # Build PLM config & tokenizer
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=1,
        finetuning_task="stsb"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
    )
    logger.info(tokenizer)


    # Build data loader
    datasets = load_dataset("glue", "stsb")
    sts_dataloader = STSDataLoader(tokenizer)
    train_loader = sts_dataloader.get_dataloader(
        data=list(datasets['train']),
        batch_size=args.train_batch_size,
    )
    valid_loader = sts_dataloader.get_dataloader(
        data=list(datasets['validation']),
        batch_size=args.valid_batch_size,
    )

    # Build model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Build optimizer
    total_train_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(args.warmup_proportion * total_train_steps)
    optimizer, scheduler = build_adamw_optimizer(args.lr,
                                                 model,
                                                 total_train_steps,
                                                 num_warmup_steps)

    # Set metric
    metric = load_metric("glue", "stsb")

    # Build trainer
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      train_loader=train_loader,
                      valid_loader=valid_loader,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      metric=metric,
                      device=device,
                      logger=logger)

    logger.info("***** training start *****")
    logger.info("Learning rate: " + f"{args.lr}")
    logger.info(f"Batch_size : {args.train_batch_size * max(num_gpus, 1)}")

    best_score = 0.0
    for epoch in range(args.epochs):
        trainer.train(epoch, args.log_interval)
        best_score = trainer.eval(best_score, args.model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default='bert-base-uncased',
        help='Path to pretrained model or model name from huggingface'
    )
    parser.add_argument(
        '--train_batch_size', type=int, default=16,
        help='input batch size for training'
    )
    parser.add_argument(
        '--valid_batch_size', type=int, default=32,
        help='input batch size for validing'
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='number of epochs to train'
    )
    parser.add_argument(
        '--lr', type=float, default=5e-5,
        help='learning rate'
    )
    parser.add_argument(
        '--warmup_proportion', type=float, default=0.1,
        help="Proportion of lr increasing steps"
    )
    parser.add_argument(
        '--seed', type=int, default=1,
        help='random seed'
    )
    parser.add_argument(
        '--log_interval', type=int, default=100,
        help='how many batches to wait before logging training status'
    )

    # Container environment
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get('SM_MODEL_DIR', './model'),
        help='path to save output'
    )
    args = parser.parse_args()

    args.model_name_or_path = 'roberta-base'

    train(args)
