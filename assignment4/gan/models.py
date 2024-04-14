import torch
import torch.nn as nn

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3, layernorm=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, out_channels=128, kernel_size=4, stride=2, padding=1)
        if layernorm == False:
            self.conv2 = torch.nn.Sequential(
                nn.Conv2d(128, out_channels=256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
            )
            self.conv3 = torch.nn.Sequential(
                nn.Conv2d(256, out_channels=512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
            )
            self.conv4 = torch.nn.Sequential(
                nn.Conv2d(512, out_channels=1024, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2),
            )
        else:
            self.conv2 = torch.nn.Sequential(
                nn.Conv2d(128, out_channels=256, kernel_size=4, stride=2, padding=1),
                nn.LayerNorm(256),
                nn.LeakyReLU(0.2),
            )
            self.conv3 = torch.nn.Sequential(
                nn.Conv2d(256, out_channels=512, kernel_size=4, stride=2, padding=1),
                nn.LayerNorm(512),
                nn.LeakyReLU(0.2),
            )
            self.conv4 = torch.nn.Sequential(
                nn.Conv2d(512, out_channels=1024, kernel_size=4, stride=2, padding=1),
                nn.LayerNorm(1024),
                nn.LeakyReLU(0.2),
            )
        self.conv5 = torch.nn.Conv2d(1024, out_channels=1, kernel_size=4, stride=1, padding=0)
        self.relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
      
    def forward(self, x):

        x = self.relu(self.conv1(x))
 
        x = self.conv2(x)
  
        x = self.conv3(x)

        x = self.conv4(x)
     
        x = self.relu(self.conv5(x))
     
        x = x.view(x.shape[0], -1)
 
        # x = self.sigmoid(x)
   
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3, layernorm=False):
        super().__init__()
        self.noise_dim = noise_dim
        if layernorm == False:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(noise_dim, out_channels=1024, kernel_size=4, stride=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            )
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(1024, out_channels=512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
            self.deconv3 = nn.Sequential(
                nn.ConvTranspose2d(512, out_channels=256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
            self.deconv4 = nn.Sequential(
                nn.ConvTranspose2d(256, out_channels=128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        else:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(noise_dim, out_channels=1024, kernel_size=4, stride=1),
                nn.LayerNorm(1024),
                nn.ReLU(),
            )
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(1024, out_channels=512, kernel_size=4, stride=2, padding=1),
                nn.LayerNorm(512),
                nn.ReLU(),
            )
            self.deconv3 = nn.Sequential(
                nn.ConvTranspose2d(512, out_channels=256, kernel_size=4, stride=2, padding=1),
                nn.LayerNorm(256),
                nn.ReLU(),
            )
            self.deconv4 = nn.Sequential(
                nn.ConvTranspose2d(256, out_channels=128, kernel_size=4, stride=2, padding=1),
                nn.LayerNorm(128),
                nn.ReLU(),
            )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        return x

def apply_spectral_norm(module):
    for child_name, child in module.named_children():
        if isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        # Apply spectral normalization
            setattr(module, child_name, spectral_norm(child))
        else:
        # Recursively apply to child modules
            apply_spectral_norm(child)
            
def check_spectral_norm(module, module_name=""):
  for name, submodule in module.named_children():
    full_name = module_name + ('.' if module_name else '') + name
      # Check if the spectral normalization attributes are present
    if hasattr(submodule, 'weight_u'):
      print(f"{full_name}: Spectral Norm Applied - True")
    else:
      print(f"{full_name}: Spectral Norm Applied - False")
    check_spectral_norm(submodule, full_name)
