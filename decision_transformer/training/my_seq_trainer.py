 # ここで損失関数を出して、逆伝播させていると思う

import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class MySequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, modes, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)
        mode_target = torch.clone(modes)

        state_preds, action_preds, mode_preds, return_preds = self.model.forward(
            states, actions, rewards, modes, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        mode_dim = mode_preds.shape[2]
        mode_preds = mode_preds.reshape(-1, mode_dim)[attention_mask.reshape(-1) > 0]
        mode_target = mode_target.reshape(-1, mode_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None, mode_preds,
            None, action_target, None, mode_target,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
