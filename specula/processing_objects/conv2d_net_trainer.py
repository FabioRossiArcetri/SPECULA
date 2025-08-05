import os

from scipy.stats.tests.test_continuous_fit_censored import optimizer
import torch
import torch.nn as nn
import torch.optim as optim
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue

from specula.base_value import BaseValue

import torch
import torch.nn as nn
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = nn.Identity()
        if in_channels != out_channels or downsample:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)

class Conv2dResNet(nn.Module):
    def __init__(self,
                 input_size=64,
                 input_channels=1,
                 base_channels=32,
                 num_blocks=4,
                 output_size=5):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        layers = []
        in_channels = base_channels
        out_channels = base_channels
        current_size = input_size

        for i in range(num_blocks):
            # Downsample if current spatial size > 4x4
            downsample = current_size > 4
            out_channels = in_channels * 2 if downsample else in_channels

            layers.append(ResidualBlock(in_channels, out_channels, downsample=downsample))

            if downsample:
                current_size = math.ceil(current_size / 2)
            in_channels = out_channels

        self.res_blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(in_channels // 2, output_size)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.pool(x)
        return self.regressor(x)


class Conv2dNet(nn.Module):
    def __init__(self, hidden_channels=64, output_size=6):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.05),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.05),
            nn.Linear(hidden_channels, output_size)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.regressor(x)
    

class Conv2dNetTrainer(BaseProcessingObj):
    def __init__(self,
                 network_filename,
                 nmodes = 5,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.verbose = True
        self.network_filename = network_filename
        self.nmodes = nmodes
        
        self.device = torch.device("cuda")

        #self.model = Conv2dNet(            
        #    hidden_channels= 64,
        #    output_size=self.nmodes).to(self.device)

        self.model = Conv2dResNet(
                input_size=160,
                 input_channels=1,
                 base_channels=128,
                 num_blocks=4,
                 output_size=self.nmodes).to(self.device)
        
        self.loss_fn = nn.MSELoss()        

        self.optimizer = optim.Adam( self.model.parameters(), lr=1e-5 )
        
        self.inputs['input_2d_batch'] = InputValue(type=BaseValue)
        self.inputs['labels'] = InputValue(type=BaseValue)

        self.outputs["loss"] = None
        self.X = None
        self.y = None
        self.min_loss = 1e9
        self.firstTrigger = True
        
    def rescale(self, v):        
        mean = self.xp.mean(v)
        v -= mean
        std = self.xp.std(v)
        v /= std
        return v, mean, std

    def trigger(self):
        self.X = self.local_inputs['input_2d_batch']
        self.y = self.local_inputs['labels']        

        if self.X is None or self.y is None:
            return
        if self.X.generation_time < self.current_time or self.y.generation_time < self.current_time:
            return

        ph = self.X.get_value()[:, 1]*self.X.get_value()[:, 0]

        modes = self.y.get_value()[:, 1:self.nmodes+1]        
        
        if self.firstTrigger:
            ph, self.meanp, self.stdp = self.rescale(ph)
            modes, self.meanmodes, self.stdmodes = self.rescale(modes)
            self.firstTrigger = False
        else:
            # ph -= self.meanp
            ph /= self.stdp
            # modes -= self.meanmodes
            modes /= self.stdmodes

        ph = ph[:, self.xp.newaxis, :, :]  # Ensure shape (B, 1, H, H)

        inputs = ( torch.tensor(ph, dtype=torch.float32, device=self.device) )
        targets = ( torch.tensor(modes, dtype=torch.float32, device=self.device))        
        
        # === Training step ===
        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model(inputs)
        self.loss = self.loss_fn(preds, targets)
        self.loss.backward()        
        self.optimizer.step()
        # === Evaluation step ===
        self.model.eval()
        with torch.no_grad():
            eval_preds = self.model(inputs)
            eval_loss = self.loss_fn(eval_preds, targets).item()            
        if eval_loss < self.min_loss:            
            self.min_loss = eval_loss
            torch.save(self.model, self.network_filename)
            if self.verbose:
                print(f"[{self.name}] Model saved to {self.network_filename}")
        if self.verbose:
            print(f"[{self.name}] Training loss: {self.loss.item():.6f} | Eval loss: {eval_loss:.6f}")
        
        
    def post_trigger(self):
        super().post_trigger()
        self.outputs["loss"] = self.loss.item()


    def finalize(self):
        print('self.min_loss', self.min_loss)
        print(f"{self.meanp}")
        print(f"{self.stdp}")
        print(f"{self.meanmodes}")
        print(f"{self.stdmodes}")

