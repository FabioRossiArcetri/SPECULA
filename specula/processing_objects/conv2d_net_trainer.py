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
import torch.nn.functional as F

class UNetRegressor(nn.Module):
    def __init__(self, input_channels=1, output_size=5, base_channels=32, input_size=(64, 64)):
        super().__init__()

        # Encoder
        self.enc1 = EnhancedConvBlock(input_channels, base_channels)
        self.pool1 = nn.AvgPool2d(2)
        self.enc2 = EnhancedConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.AvgPool2d(2)
        self.enc3 = EnhancedConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.AvgPool2d(2)
        self.enc4 = EnhancedConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.AvgPool2d(2)
        self.enc5 = EnhancedConvBlock(base_channels * 8, base_channels * 16)
        self.pool5 = nn.AvgPool2d(2)

        self.bottleneck = EnhancedConvBlock(base_channels * 16, base_channels * 32)

        # Decoder
        self.up5 = nn.ConvTranspose2d(base_channels * 32, base_channels * 16, kernel_size=2, stride=2)
        self.att5 = AttentionGate(base_channels * 16, base_channels * 16, base_channels * 8)
        self.dec5 = EnhancedConvBlock(base_channels * 32, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(base_channels * 8, base_channels * 8, base_channels * 4)
        self.dec4 = EnhancedConvBlock(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(base_channels * 4, base_channels * 4, base_channels * 2)
        self.dec3 = EnhancedConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(base_channels * 2, base_channels * 2, base_channels)
        self.dec2 = EnhancedConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.att1 = AttentionGate(base_channels, base_channels, base_channels // 2)
        self.dec1 = EnhancedConvBlock(base_channels * 2, base_channels)

        # Multi-scale fusion
        self.ms_bottleneck = MultiScaleFeatureExtractor(base_channels * 36)
        self.ms_dec5 = MultiScaleFeatureExtractor(base_channels * 16)
        self.ms_dec4 = MultiScaleFeatureExtractor(base_channels * 8)
        self.ms_dec3 = MultiScaleFeatureExtractor(base_channels * 4)
        self.ms_dec2 = MultiScaleFeatureExtractor(base_channels * 2)
        self.ms_dec1 = MultiScaleFeatureExtractor(base_channels)

        # 🔥 Dynamically infer feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            features = self._forward_features(dummy)
            feature_dim = features.shape[1]

        # Fully connected regressor
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.02),
            nn.Linear(128, output_size)
        )

    def _forward_features(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))

        b = self.bottleneck(self.pool5(e5))

        d5 = self.up5(b)
        d5 = torch.cat([self.att5(d5, e5), d5], dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat([self.att4(d4, e4), d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([self.att3(d3, e3), d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([self.att2(d2, e2), d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([self.att1(d1, e1), d1], dim=1)
        d1 = self.dec1(d1)

        return torch.cat([
            self.ms_bottleneck(b),
            self.ms_dec5(d5),
            self.ms_dec4(d4),
            self.ms_dec3(d3),
            self.ms_dec2(d2),
            self.ms_dec1(d1)
        ], dim=1)

    def forward(self, x):
        features = self._forward_features(x)
        return self.fc(features)

# ==============================
#   Attention and Squeeze Blocks
# ==============================

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ==============================
#   Core Convolutional Blocks
# ==============================

class EnhancedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.02, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)

        # residual adapter if channels differ
        self.residual_adapter = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else None
        )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        out = self.se(out)

        if self.use_residual:
            if self.residual_adapter is not None:
                residual = self.residual_adapter(residual)
            out += residual

        return self.relu(out)


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.pool3 = nn.AdaptiveAvgPool2d(2)
        self.pool4 = nn.AdaptiveAvgPool2d(4)
        self.pool5 = nn.AdaptiveAvgPool2d(8)

    def forward(self, x):
        B, C, H, W = x.shape
        features = []
        for pool in [self.pool1, self.pool2, self.pool3, self.pool4, self.pool5]:
            pooled = pool(x)
            if pooled.shape[2] <= H and pooled.shape[3] <= W:  # only valid scales
                features.append(pooled.view(B, -1))
        return torch.cat(features, dim=1)

    

class WeightedSMAPELoss(nn.Module):
    """
    Symmetric Weighted Mean Absolute Percentage Error (sWMAPE).
    Loss = mean( weights_i * (2 * |y_pred - y_true|) / (|y_true| + |y_pred| + eps) )

    Args:
        weights (list or torch.Tensor): weights for each output dimension.
        reduction (str): "mean" (default), "sum", or "none"
        eps (float): small constant to avoid division by zero
    """
    def __init__(self, weights=None, reduction="mean", eps=1e-6, device=None):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        if weights is not None:
            self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32).to(device))
        else:
            self.weights = None

    def forward(self, preds, targets):
        """
        preds:   (batch_size, num_outputs)
        targets: (batch_size, num_outputs)
        """
        # sMAPE formula
        numerator = torch.abs(preds - targets) * 2.0
        denominator = torch.abs(targets) + torch.abs(preds) + self.eps
        smape = numerator / denominator  # (B, D)

        # Apply weights per output dimension
        if self.weights is not None:
            smape = smape * self.weights

        # Reduction
        if self.reduction == "mean":
            return smape.mean()
        elif self.reduction == "sum":
            return smape.sum()
        return smape


class WeightedMSELoss(nn.Module):
    """Element-wise weighted MSE loss"""
    def __init__(self, weights, device):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32).to(device))

    def forward(self, preds, targets):
        diff = preds - targets
        weighted_sq = self.weights * (diff ** 2)
        return weighted_sq.mean()


class Conv2dNet(nn.Module):
    def __init__(self, hidden_channels=64, output_size=6):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.02),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.02),
            nn.Linear(hidden_channels, output_size)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.regressor(x)
    

class Conv2dNetTrainer(BaseProcessingObj):
    def __init__(self,
                 network_filename,
                 nmodes = 20,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.verbose = False
        self.network_filename = network_filename
        self.nmodes = nmodes
        
        self.device = torch.device("cuda")

        #self.model = Conv2dNet(            
        #    hidden_channels= 64,
        #    output_size=self.nmodes).to(self.device)
 
        self.model = UNetRegressor(
            input_channels=1,    
            output_size=20,
            base_channels=128, 
            input_size=(160, 160)
        ).to(self.device)


        
        # self.model = Conv2dResNet(
        #   input_size=32,
        #    input_channels=1,
        #    base_channels=16,
        #    num_blocks=8,
        #    block_depth=3,
        #    output_size=5
        # ).to(self.device)

        
        # self.loss_fn = WeightedSMAPELoss(weights=[1.0, 0.01, 0.01, 0.01, 0.01], device=self.device )  # tune per output dimension
        # self.loss_fn = WeightedMSELoss(weights=[1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], device=self.device )  # tune per output dimension
        self.loss_fn = WeightedMSELoss(weights=[1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                                1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], device=self.device )  # tune per output dimension

        #self.model = Conv2dResNet(
        #        input_size=160,
        #         input_channels=1,
        #         base_channels=128,
        #         num_blocks=4,
        #         output_size=self.nmodes).to(self.device)
        
        # self.loss_fn = nn.MSELoss()        

        self.optimizer = optim.Adam( self.model.parameters(), lr=1e-3 )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            verbose=True
        )

        # Gradient clipping threshold
        self.grad_clip_value = 1.0
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
        
        #if self.firstTrigger:
        #    ph, self.meanp, self.stdp = self.rescale(ph)
        #    modes, self.meanmodes, self.stdmodes = self.rescale(modes)
        #    self.firstTrigger = False

        ## ph -= self.meanp
        #ph /= self.stdp
        ## modes -= self.meanmodes
        #modes /= self.stdmodes

        ph = ph[:, self.xp.newaxis, :, :]  # Ensure shape (B, 1, H, H)

        inputs = ( torch.tensor(ph, dtype=torch.float32, device=self.device) )
        targets = ( torch.tensor(modes, dtype=torch.float32, device=self.device))        
        
        # === Training step ===
        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model(inputs)
        self.loss = self.loss_fn(preds, targets)
                
        self.loss.backward()        
        

        #  # 🔧 Gradient clipping
        #nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

        self.optimizer.step()

        # 🔧 LR scheduler step (based on validation loss; here we only have training loss)
        # self.scheduler.step(self.loss.item())
        
        
        # === Evaluation step ===
        self.model.eval()
        with torch.no_grad():
            eval_preds = self.model(inputs)
            eval_loss = self.loss_fn(eval_preds, targets).item()            
        if eval_loss < self.min_loss:            
            self.min_loss = eval_loss
            torch.save(self.model, self.network_filename)
            #if self.verbose:
            print(f"[{self.name}] Model saved to {self.network_filename}")
        #if self.verbose:
        print(f"[{self.name}] Training loss: {self.loss.item():.6f} | Eval loss: {eval_loss:.6f}")
        
        
    def post_trigger(self):
        super().post_trigger()
        self.outputs["loss"] = self.loss.item()


    def finalize(self):
        print('self.min_loss', self.min_loss)
        #print(f"{self.meanp}")
        #print(f"{self.stdp}")
        #print(f"{self.meanmodes}")
        #print(f"{self.stdmodes}")

