import numpy as np
import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import MyTrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


#元のDTは過去のrewardを学習させていない
#このmyDTはrewardの代わりにmodeを学習させる

class MyDecisionTransformer(MyTrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            mode_dim, # 追加
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, mode_dim, max_length=max_length) # 追加

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: このGPT2ModelとデフォルトのHuggingfaceバージョンとの唯一の違いは、位置埋め込みが削除されていることです。
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_mode = torch.nn.Linear(self.mode_dim, hidden_size) # 追加

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: 本稿では、状態や リターンの予測はしていない。
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_mode = torch.nn.Linear(hidden_size, self.mode_dim) # 追加
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, modes, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        mode_embeddings = self.embed_mode(modes) # 追加
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # 時間embeddingsは位置embeddingsと同様に扱われる
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        mode_embeddings = mode_embeddings + time_embeddings # 追加
        returns_embeddings = returns_embeddings + time_embeddings

        # これは（R_1, s_1, a_1, R_2, s_2, a_2, ...）のようになり、状態が行動を予測するので、自己回帰的な意味でうまく機能する。
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings, mode_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 4*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # アテンション・マスクをstackされたインプットに適合させるには、それをstackしなければならない。
        # 個人的解釈 stackされたインプット -> stacked_inputs
        #           それをstackしなければ -> attention_maskをstackしなければ
        #※stack((a, b, c),dim=1): 行方向にa, b, cを結合(重ねる)
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 4*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # 2番目の次元が元のリターン(0)、ステート(1)、アクション(2)に対応するようにxを再形成する。
        # すなわち、x[:,1,t]はs_tのトークンである。
        x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])  # 状態とアクションを指定して、次のリターンを予測する
        state_preds = self.predict_state(x[:,2])    # 状態とアクションが与えられたら、次の状態を予測する
        mode_preds = self.predict_mode(x[:,2]) # 状態とアクションが与えられたら、次の報酬を予測する  # 追加
        action_preds = self.predict_action(x[:,1])  # 状態から次の行動を予測する

        return state_preds, action_preds, mode_preds, return_preds

    def get_action(self, states, actions, rewards, modes, returns_to_go, timesteps, **kwargs):
        # このモデルでは過去の報酬は気にしない

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        modes = modes.reshape(1, -1, self.mode_dim)  # 追加
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            modes = modes[:,-self.max_length:] # 追加
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim), device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            modes = torch.cat( # 追加
                [torch.zeros((modes.shape[0], self.max_length - modes.shape[1], self.mode_dim), device=modes.device), modes],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, _, return_preds = self.forward(
            states, actions, None, modes, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs) # 編集 None -> modes

        return action_preds[0,-1]
