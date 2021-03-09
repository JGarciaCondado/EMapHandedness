from torch import nn

class AlphaNet(nn.Module):
    """ 3D CNN to estiamte the probability that there is an alpha helix in the center of the box.
    """
    def __init__(self):
        super().__init__()

        # Convlutional layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size = 5)
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=8, kernel_size = 5)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size = 3)
        self.conv4 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size = 3)
        self.conv5 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size = 2)

        # Max pool layer
        self.pool = nn.MaxPool3d(2)

        # Linear layers
        self.linear1 = nn.Linear(512,120)
        self.linear2 = nn.Linear(128,1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input tensor through the CNN operations
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.relu(x)
        # Flatten the tensor into a vecto
        x = x.view(-1,512)
        # Pass the tensor through the FC layes
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.simoid(x)
        return x
