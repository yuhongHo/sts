from datasets import Metric
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer


class Trainer(object):
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: LambdaLR,
                 metric: Metric,
                 device: str,
                 logger):

        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = metric

        self.device = device
        self.logger = logger

    def save_checkpoint(self, output_dir: str):
        self.logger.info("saving model..")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def optimize(self, loss: float):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def train(self, epoch: int, log_interval: int):
        self.model.train()

        train_loss = 0.0
        n_data = 0
        for i, batch in enumerate(self.train_loader, 1):
            input_ids, attention_mask, token_type_ids, labels = [x.to(self.device) for x in batch]
            # token_type_ids = token_type_ids.unsqueeze(1)
            output = self.model(
                input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels
            )
            loss = output['loss']
            self.optimize(loss)

            train_loss += loss * input_ids.size(0)
            n_data += input_ids.size(0)

            if i % log_interval == 0 or i == len(self.train_loader) - 1:
                self.logger.info(f"[Epoch {epoch+1} / Step {i}] " + "loss={:.4f}".format(train_loss / n_data))
                train_loss = 0.0
                n_data = 0

    @torch.no_grad()
    def eval(self, best_score: float, output_dir: str):
        self.model.eval()
        # token_type_ids = token_type_ids.unsqueeze(1)
        valid_loss = 0.0
        for _, batch in enumerate(self.valid_loader, 1):
            input_ids, attention_mask, token_type_ids, labels = [x.to(self.device) for x in batch]
            
            output = self.model(
                input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels
            )
            valid_loss += output['loss'] * input_ids.size(0)
            logits = output['logits']

            self.metric.add_batch(predictions=logits, references=labels)

        spearmanr = self.metric.compute()['spearmanr']
        valid_loss /= len(self.valid_loader.dataset)

        self.logger.info("[Valid] " + "loss={:.4f}  spearmanr={:.4f};".format(valid_loss, spearmanr))
        if spearmanr > best_score:
            best_score = spearmanr
            self.logger.info("Hit the best score")
            self.save_checkpoint(output_dir)
        return best_score
