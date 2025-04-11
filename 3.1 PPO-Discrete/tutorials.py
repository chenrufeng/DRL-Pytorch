


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 假设我们已经有了策略网络 PolicyNet 和价值网络 ValueNet
# PolicyNet 输出动作的概率分布
# ValueNet 输出状态的价值估计

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Linear(state_dim, action_dim) # 简化模型，实际应用中可以使用更复杂的网络结构

    def forward(self, state):
        logits = self.fc(state)
        probs = torch.softmax(logits, dim=-1) # 输出动作概率分布
        return probs

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc = nn.Linear(state_dim, 1) # 简化模型

    def forward(self, state):
        value = self.fc(state)
        return value

# 超参数
clip_param = 0.2
ppo_epoch = 10
mini_batch_size = 64
learning_rate_actor = 3e-4
learning_rate_critic = 1e-3
gamma = 0.99
gae_lambda = 0.95

# 假设我们已经收集到了一批数据，存储在 buffers 中
# buffers 包含：states, actions, rewards, log_probs (旧策略的动作对数概率), values (旧策略的价值估计)

def ppo_update(policy_net, value_net, optimizer_actor, optimizer_critic, buffers):
    states = torch.tensor(buffers['states'], dtype=torch.float)
    actions = torch.tensor(buffers['actions'], dtype=torch.long)
    rewards = torch.tensor(buffers['rewards'], dtype=torch.float)
    old_log_probs = torch.tensor(buffers['log_probs'], dtype=torch.float)
    old_values = torch.tensor(buffers['values'], dtype=torch.float)

    # 计算优势函数 (Generalized Advantage Estimation, GAE)  # At = δt + (λγ)At_1
    values = value_net(states).squeeze(-1)
    advantages = torch.zeros_like(rewards)
    last_gae_lambda = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - (t == len(rewards) - 1)) - values[t] # 假设 values 长度比 rewards 长 1，最后一个 value 为 0
        advantages[t] = last_gae_lambda = delta + gamma * gae_lambda * (1 - (t == len(rewards) - 1)) * last_gae_lambda
    returns = advantages + values[:-1] # returns 长度和 rewards 一致
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # 优势函数标准化
    
    # PPO 迭代更新
    for _ in range(ppo_epoch):
        # Mini-batch 采样 (这里简化了，实际应用中需要更完善的 mini-batch 采样)
        for start in range(0, len(states), mini_batch_size):
            end = start + mini_batch_size
            mini_batch_states = states[start:end]
            mini_batch_actions = actions[start:end]
            mini_batch_advantages = advantages[start:end]
            mini_batch_returns = returns[start:end]
            mini_batch_old_log_probs = old_log_probs[start:end]
            mini_batch_old_values = old_values[start:end]

            # 计算新策略的动作概率和对数概率
            new_probs = policy_net(mini_batch_states)
            new_log_probs = new_probs.log_prob(mini_batch_actions)

            # 计算重要性采样率
            ratio = torch.exp(new_log_probs - mini_batch_old_log_probs)

            # 计算裁剪后的目标函数
            surr1 = ratio * mini_batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * mini_batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean() # 负号是因为我们要最大化目标函数，而优化器是最小化损失函数

            # 计算价值函数损失
            new_values = value_net(mini_batch_states).squeeze(-1)
            critic_loss = nn.MSELoss()(new_values, mini_batch_returns)

            # 更新策略网络和价值网络
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

    return actor_loss.item(), critic_loss.item()

# 初始化策略网络、价值网络和优化器 (假设 state_dim 和 action_dim 已定义)
policy_net = PolicyNet(state_dim, action_dim)
value_net = ValueNet(state_dim)
optimizer_actor = optim.Adam(policy_net.parameters(), lr=learning_rate_actor)
optimizer_critic = optim.Adam(value_net.parameters(), lr=learning_rate_critic)

# ... (数据收集过程) ...

# 进行 PPO 更新
actor_loss, critic_loss = ppo_update(policy_net, value_net, optimizer_actor, optimizer_critic, buffers)

print("Actor Loss:", actor_loss)
print("Critic Loss:", critic_loss)
