import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import os


# class Linear_QNet(nn.Module):
#     def __init__(self, input_size, hidden_sizes, output_size):
#         super().__init__()
#         self.layer1 = nn.Linear(input_size, hidden_sizes[0])
#         self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
#         self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
#         self.output_layer = nn.Linear(hidden_sizes[2], output_size)

#     def forward(self, x):
#         x = f.relu(self.layer1(x))
#         x = f.relu(self.layer2(x))
#         x = f.relu(self.layer3(x))
#         x = self.output_layer(x)
#         return x

class CNN_QNet(nn.Module):
    def __init__(self, output_size=1, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # After conv layers: 64 channels * 20 rows * 10 cols
        self.fc1 = nn.Linear(64 * 20 * 10, 256)
        self.fc2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, output_size)
    
    def forward(self, x):
        # Input x shape: (batch, 1, 20, 10)
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        return self.output_layer(x)

class QTrainer:
    def __init__(self, model1, model2, lr, gamma, memory, EPOCH, BATCH_SIZE):
        self.memory = memory
        self.lr = lr
        self.gamma = gamma
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.model1 = model1
        self.model2 = model2
        self.optimizer1 = optim.Adam(model1.parameters(), lr=self.lr)
        #self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()
        self.q_values = []

    def update_lr(self, new_lr):
        for param_group in self.optimizer1.param_groups:
            param_group['lr'] = new_lr

    def fit(self, x, y):
        x_train = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
        y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1) if not isinstance(y, torch.Tensor) else y

        self.optimizer1.zero_grad()
        outputs = self.model1(x_train)
        loss = self.criterion(outputs, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model1.parameters(), max_norm=1.0)
        self.optimizer1.step()

        return loss.item()

    def train_step(self):
        if self.BATCH_SIZE > len(self.memory):
            return 0

        # Sample batch
        batch, indices, weights = self.memory.sample(self.BATCH_SIZE)
        states, next_states, rewards, dones = zip(*batch)

        # Convert to tensors for batch processing
        # state_batch = torch.tensor(states, dtype=torch.float32)
        # next_state_batch = torch.tensor(next_states, dtype=torch.float32)

        # Add channel dimension for CNN
        state_batch = torch.tensor(states, dtype=torch.float32).unsqueeze(1)
        # Shape: (batch, 1, 20, 10)
        next_state_batch = torch.tensor(next_states, dtype=torch.float32).unsqueeze(1)

        reward_batch = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
        done_batch = torch.tensor(dones, dtype=torch.bool)

        # Predict Q-values
        current_q_values = self.model1(state_batch)

        # Target network evaluation for next Q-values
        next_q_values_target = self.model2(next_state_batch)
        next_q_values_primary = self.model1(next_state_batch)

        # Compute target Q-values using Double DQN
        max_actions = torch.argmax(next_q_values_primary, dim=1)
        target_q_values = reward_batch + (1 - done_batch.float().view(-1, 1)) * self.gamma * next_q_values_target.gather(1,max_actions.view(-1,1))

        # Compute TD error for prioritized experience replay
        td_errors = torch.abs(target_q_values - current_q_values.gather(1, max_actions.view(-1, 1)))

        # Update priorities in memory (use the entire batch of td_errors at once)
        self.memory.update_priority(indices, td_errors)

        # Train the primary model
        return self.fit(state_batch, target_q_values)

    def train(self):
        total_loss = 0
        for epoch in range(self.EPOCH):
            loss = self.train_step()
            total_loss += loss
        return total_loss