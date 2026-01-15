from transformers import Trainer
import torch
class LaViTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if num_items_in_batch is not None:
             outputs = model(**inputs, num_items_in_batch=num_items_in_batch)
        else:
             outputs = model(**inputs)
        total_loss = outputs.loss
        return (total_loss, outputs) if return_outputs else total_loss