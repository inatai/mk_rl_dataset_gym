 # ここで損失関数を出して、逆伝播させていると思う

import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # print(torch.mean(rtg))
        action_target = torch.clone(actions)

        # reward = [batch_size, seq_length(20), 1]
        # rtg = [batch_size, seq_length+1(21), 1]

        # print("################")
        # print(self.batch_size)
        # print(rewards[0])
        # print(rtg[0])
        # print(rtg[:,:-1])
        # print(rewards[0].shape)
        # print(rtg[0].shape)
        # print(rtg[:,:-1].shape)

        # for rew, rtg in zip(rewards[0], rtg[0,:,-1]):
        #     print(f'{rew} / {rtg}')


        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
