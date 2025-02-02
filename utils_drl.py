from typing import (
    Optional,
)

import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import DQN, Dueling_DQN


class Agent(object):

    def __init__(
            self,
            action_dim: int,
            device: TorchDevice,
            gamma: float,
            seed: int,

            eps_start: float,
            eps_final: float,
            eps_decay: float,

            restore: Optional[str] = None,
            
            use_dueling = False,
            use_DDQN = False,
            use_PR = False, #Prioritized Experience Replay
    ) -> None:
        self.__action_dim = action_dim
        self.__device = device
        self.__gamma = gamma

        self.__eps_start = eps_start
        self.__eps_final = eps_final
        self.__eps_decay = eps_decay

        self.__eps = eps_start
        self.__r = random.Random()
        self.__r.seed(seed)
        
        self.use_dueling = use_dueling
        self.use_DDQN = use_DDQN
        self.use_PR = use_PR

        if not use_dueling:
            self.__policy = DQN(action_dim, device).to(device)
            self.__target = DQN(action_dim, device).to(device)
        else:
            self.__policy = Dueling_DQN(action_dim, device).to(device)
            self.__target = Dueling_DQN(action_dim, device).to(device)
            
        if restore is None:
            self.__policy.apply(self.__policy.init_weights)
        else:
            self.__policy.load_state_dict(torch.load(restore))
        self.__target.load_state_dict(self.__policy.state_dict())
        self.__optimizer = optim.Adam(
            self.__policy.parameters(),
            lr=0.0000625,
            eps=1.5e-4,
        )
        self.__target.eval()

    def run(self, state: TensorStack4, training: bool = False) -> int:
        """run suggests an action for the given state."""
        if training:
            self.__eps -= \
                (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)

        if self.__r.random() > self.__eps:
            with torch.no_grad():
                return self.__policy(state).max(1).indices.item()
        return self.__r.randint(0, self.__action_dim - 1)
    
    
    def learn(self, memory: ReplayMemory, batch_size: int) -> float:
        """learn trains the value network via TD-learning."""
        if self.use_PR:
            state_batch, action_batch, reward_batch, next_batch, done_batch, idxs, ISWeights = \
                memory.sample(batch_size)
        else:
            state_batch, action_batch, reward_batch, next_batch, done_batch = \
                memory.sample(batch_size)
            
        if self.use_DDQN:
            actions_value = self.__policy(next_batch.float())
            max_val_action = actions_value.max(1)[1].unsqueeze(-1)

            actions_value = self.__target(next_batch.float()).detach()  
            expected = reward_batch + (self.__gamma * actions_value.gather(1, max_val_action)) * (1. - done_batch) 
            values = self.__policy(state_batch.float()).gather(1, action_batch)
        else:
            values = self.__policy(state_batch.float()).gather(1, action_batch)
            values_next = self.__target(next_batch.float()).max(1).values.detach()
            expected = (self.__gamma * values_next.unsqueeze(1)) * \
                (1. - done_batch) + reward_batch
            
            
        if self.use_PR:    
            abs_errors = torch.abs(expected - values).data.cpu().numpy()
            # update priority
            memory.batch_update(idxs, abs_errors)
            loss = (ISWeights * F.smooth_l1_loss(values, expected, reduction = 'none')).mean()
        else:
            loss = F.smooth_l1_loss(values, expected)

        self.__optimizer.zero_grad()
        loss.backward()
        #for param in self.__policy.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()

        return loss.item()


    def sync(self) -> None:
        """sync synchronizes the weights from the policy network to the target
        network."""
        self.__target.load_state_dict(self.__policy.state_dict())

    def save(self, path: str) -> None:
        """save saves the state dict of the policy network."""
        torch.save(self.__policy.state_dict(), path)
