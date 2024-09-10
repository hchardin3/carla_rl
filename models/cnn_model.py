import torch as T
import torch.nn as nn
import torch.optim as optim
import os

class CNN_DeepQNetwork(nn.Module):
    """
    A deep Q-Network (DQN) model for reinforcement learning.
    """
    def __init__(self, action_size: int, hidden_size: int = 64, learning_rate: float = 0.01):
        super(CNN_DeepQNetwork, self).__init__()

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        
        self.vision_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.to(self.device)
        self.conv_output_size = self._get_conv_output_size((3, 84, 84))

        self.decision_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9)
        self.loss_function = nn.SmoothL1Loss()  # Huber loss for stability

        self.to(self.device)
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        if x.device != self.device:
            x = x.to(self.device)
        
        x = self.vision_layer(x)
        x = self.decision_layer(x)

        return x
    
    def _get_conv_output_size(self, input_shape):
        """
        Computes the size of the output of the convolutional layers.
        This is required to correctly size the first fully connected layer.
        """
        # Create a dummy input tensor with the shape of a typical input
        dummy_input = T.zeros(1, *input_shape).to(self.device)  # Batch size is 1
        output_feat = self.vision_layer(dummy_input)
        return int(T.prod(T.tensor(output_feat.size()[1:])))

    
    def save(self, file_name:str='model.pth'):
        """
        Save the model state dictionary and optimizer state.
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        T.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, file_name)
    
    def load(self, folder_path:str='./model', file_name:str='model.pth'):
        """
        Load the state dictionary and optimizer state for this model.
        """
        file_path = os.path.join(folder_path, file_name)
        checkpoint = T.load(file_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
