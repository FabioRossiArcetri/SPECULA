import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================
#   Helper Function for GroupNorm
# ==============================

def get_num_groups(channels, max_groups=32):
    """
    Find the largest valid number of groups for GroupNorm.
    The number of groups must divide the number of channels evenly.
    """
    for groups in [max_groups, 16, 8, 4, 2, 1]:
        if channels % groups == 0:
            return groups
    return 1

'''

(base12) frossi@gandalf:~/dev/SPECULA/config$ tail -f  out2.log | grep loss
  Best loss: 0.000110

(base12) frossi@gandalf:~/dev/SPECULA/config$ tail -f  out3.log | grep loss
  Best loss: 0.000415

(base12) frossi@gandalf:~/dev/SPECULA/config$ tail -f  out4.log | grep loss
  Best loss: 0.000388

(base12) frossi@gandalf:~/dev/SPECULA/config$ tail -f  out1.log | grep loss
  Best loss: 0.000278


'''

class UNetRegressor(nn.Module):
    """
    Improved U-Net regressor with:
    - Efficient architecture with fewer parameters
    - Modern activation functions (GELU)
    - Proper normalization (GroupNorm for small batches)
    - Multi-scale feature extraction
    - Attention mechanisms at bottleneck
    """
    def __init__(self, input_channels=1, output_size=5, base_channels=32, 
                 input_size=(64, 64), dropout_level=0.1, depth=5, conv_block_type = 0):
        super().__init__()
        
        self.dropout_level = dropout_level
        self.depth = depth
        
        # Build encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        if conv_block_type==0:                              # __ @ 32ch
            self.aConvBlock = ConvBlock
        elif conv_block_type==1:                            # 0.0246 () @ 32ch
            self.aConvBlock = DepthwiseSeparableConvBlock
        elif conv_block_type==2:                            # 0.0081 (step 718) @ 32ch
            self.aConvBlock = InvertedResidualConvBlock
        elif conv_block_type==3:                            # 0.0229 @ 32ch
            self.aConvBlock = CoordAttentionConvBlock   
        elif conv_block_type==4:                            # 0.0368... @ 32ch
            self.aConvBlock = DenseConvBlock
        
        in_ch = input_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(self.aConvBlock(in_ch, out_ch, dropout=dropout_level))
            self.pools.append(nn.AvgPool2d(2))
            in_ch = out_ch
        
        # Bottleneck with self-attention
        bottleneck_ch = base_channels * (2 ** depth)
        self.bottleneck = nn.Sequential(
            self.aConvBlock(in_ch, bottleneck_ch, dropout=dropout_level),
            SelfAttention(bottleneck_ch),
            self.aConvBlock(bottleneck_ch, bottleneck_ch, dropout=dropout_level)
        )
        
        # Build decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(depth):
            # Input channels: bottleneck on first iteration, then decoder output
            in_ch = bottleneck_ch if i == 0 else base_channels * (2 ** (depth - i))
            skip_ch = base_channels * (2 ** (depth - i - 1))
            out_ch = base_channels * (2 ** (depth - i - 1))
            
            self.ups.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.decoders.append(self.aConvBlock(out_ch + skip_ch, out_ch, dropout=dropout_level))
        
        # Multi-scale feature extraction
        self.feature_extractors = nn.ModuleList([
            AdaptiveFeatureExtractor(bottleneck_ch)
        ] + [
            AdaptiveFeatureExtractor(base_channels * (2 ** (depth - i - 1))) 
            for i in range(depth)
        ])
        
        # Compute feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            features = self._forward_features(dummy)
            feature_dim = features.shape[1]
        
        # Regression head with skip connections
        self.regressor = RegressionHead(feature_dim, output_size, dropout_level)
    
    def _forward_features(self, x):
        # Encoder
        enc_features = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            enc_features.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        all_features = [self.feature_extractors[0](x)]
        
        # Decoder
        for i, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            x = up(x)
            skip = enc_features[-(i + 1)]
            
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
            all_features.append(self.feature_extractors[i + 1](x))
        
        return torch.cat(all_features, dim=1)
    
    def forward(self, x):
        features = self._forward_features(x)
        return self.regressor(features)


# ==============================
#   Core Building Blocks
# ==============================

# ==============================
#   Option 1: Depthwise Separable Convolutions
#   Benefits: 5-10x fewer parameters, faster training, similar accuracy
# ==============================

class DepthwiseSeparableConvBlock(nn.Module):
    """
    Uses depthwise separable convolutions for efficiency.
    Great for limited compute while maintaining performance.
    """
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        
        # First depthwise separable conv
        self.dw_conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw_conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout2d(dropout)
        
        # Second depthwise separable conv
        self.dw_conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.pw_conv2 = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout2d(dropout)
        
        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch)
        ) if in_ch != out_ch else nn.Identity()
        
        # SE attention
        self.se = SEBlock(out_ch)
    
    def forward(self, x):
        identity = self.residual(x)
        
        # First conv
        out = self.dw_conv1(x)
        out = self.pw_conv1(out)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop1(out)
        
        # Second conv
        out = self.dw_conv2(out)
        out = self.pw_conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.drop2(out)
        
        return self.se(out + identity)


# ==============================
#   Option 2: Inverted Residual Block (MobileNetV2 style)
#   Benefits: Better gradient flow, captures more features
# ==============================

class InvertedResidualConvBlock(nn.Module):
    """
    Expansion -> Depthwise -> Projection pattern.
    More expressive than standard convolutions.
    """
    def __init__(self, in_ch, out_ch, dropout=0.1, expansion=4):
        super().__init__()
        
        hidden_dim = in_ch * expansion
        
        # Expansion phase
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.GELU()
        )
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        
        # Projection phase
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch)
        )
        
        # Residual
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch)
        ) if in_ch != out_ch else nn.Identity()
        
        # SE block
        self.se = SEBlock(out_ch)
        self.final_act = nn.GELU()
    
    def forward(self, x):
        identity = self.residual(x)
        
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        
        out = self.se(out + identity)
        return self.final_act(out)


# ==============================
#   Option 3: Dense Connections (DenseNet style)
#   Benefits: Better feature reuse, stronger gradients
# ==============================

class DenseConvBlock(nn.Module):
    """
    Dense connections within the block for better feature reuse.
    """
    def __init__(self, in_ch, out_ch, dropout=0.1, growth_rate=32):
        super().__init__()
        
        # Helper function to get valid number of groups
        def get_num_groups(channels):
            # Find the largest divisor of channels that's <= 32
            for groups in [32, 16, 8, 4, 2, 1]:
                if channels % groups == 0:
                    return groups
            return 1
        
        # First layer
        self.conv1 = nn.Sequential(
            nn.GroupNorm(get_num_groups(in_ch), in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, growth_rate, 3, padding=1, bias=False),
            nn.Dropout2d(dropout)
        )
        
        # Second layer (concatenates with input)
        concat_ch1 = in_ch + growth_rate
        self.conv2 = nn.Sequential(
            nn.GroupNorm(get_num_groups(concat_ch1), concat_ch1),
            nn.GELU(),
            nn.Conv2d(concat_ch1, growth_rate, 3, padding=1, bias=False),
            nn.Dropout2d(dropout)
        )
        
        # Transition to output channels
        concat_ch2 = in_ch + 2 * growth_rate
        self.transition = nn.Sequential(
            nn.Conv2d(concat_ch2, out_ch, 1, bias=False),
            nn.GroupNorm(get_num_groups(out_ch), out_ch)
        )
        
        # Residual
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(get_num_groups(out_ch), out_ch)
        ) if in_ch != out_ch else nn.Identity()
        
        self.se = SEBlock(out_ch)
    
    def forward(self, x):
        identity = self.residual(x)
        
        # Dense connections
        out1 = self.conv1(x)
        out2 = self.conv2(torch.cat([x, out1], dim=1))
        
        # Concatenate all and transition
        out = self.transition(torch.cat([x, out1, out2], dim=1))
        
        return self.se(out + identity)

# ==============================
#   Option 4: Multi-Scale Convolutions (Inception style)
#   Benefits: Captures features at multiple scales
# ==============================

class MultiScaleConvBlock(nn.Module):
    """
    Parallel convolutions with different kernel sizes.
    """
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        
        branch_ch = out_ch // 4
        
        # 1x1 branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            nn.GroupNorm(min(32, branch_ch), branch_ch),
            nn.GELU()
        )
        
        # 3x3 branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            nn.GroupNorm(min(32, branch_ch), branch_ch),
            nn.GELU(),
            nn.Conv2d(branch_ch, branch_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, branch_ch), branch_ch),
            nn.GELU()
        )
        
        # 5x5 branch (using two 3x3)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            nn.GroupNorm(min(32, branch_ch), branch_ch),
            nn.GELU(),
            nn.Conv2d(branch_ch, branch_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, branch_ch), branch_ch),
            nn.GELU(),
            nn.Conv2d(branch_ch, branch_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, branch_ch), branch_ch),
            nn.GELU()
        )
        
        # Pool branch
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            nn.GroupNorm(min(32, branch_ch), branch_ch),
            nn.GELU()
        )
        
        self.dropout = nn.Dropout2d(dropout)
        
        # Residual
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch)
        ) if in_ch != out_ch else nn.Identity()
        
        self.se = SEBlock(out_ch)
    
    def forward(self, x):
        identity = self.residual(x)
        
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.dropout(out)
        
        return self.se(out + identity)


# ==============================
#   Option 5: Coordinate Attention Block
#   Benefits: Better spatial awareness, position encoding
# ==============================

class CoordAttentionConvBlock(nn.Module):
    """
    Adds coordinate attention for better spatial feature learning.
    """
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        
        # Coordinate attention
        self.coord_att = CoordinateAttention(out_ch)
        
        # Residual
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch)
        ) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        identity = self.residual(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.coord_att(out)
        
        return out + identity


class CoordinateAttention(nn.Module):
    """Coordinate attention mechanism"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        
        reduced = max(channels // reduction, 8)
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1 = nn.Conv2d(channels, reduced, 1, bias=False)
        self.bn1 = nn.GroupNorm(min(8, reduced), reduced)
        self.act = nn.GELU()
        
        self.conv_h = nn.Conv2d(reduced, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(reduced, channels, 1, bias=False)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Encode along H and W axes
        x_h = self.pool_h(x)  # B, C, H, 1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # B, C, W, 1
        
        # Concatenate and process
        y = torch.cat([x_h, x_w], dim=2)  # B, C, H+W, 1
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split and apply attention
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return x * a_h * a_w


# ==============================
#   Helper: SE Block (reused above)
# ==============================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(channels // reduction, 8)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.GELU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c = x.shape[:2]
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
class ConvBlock(nn.Module):
    """Modern conv block with GroupNorm and GELU activation"""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        
        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch)
        ) if in_ch != out_ch else nn.Identity()
        
        # Channel attention
        self.se = SEBlock(out_ch)
    
    def forward(self, x):
        return self.se(self.block(x) + self.residual(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(channels // reduction, 8)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.GELU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c = x.shape[:2]
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SelfAttention(nn.Module):
    """Lightweight self-attention for bottleneck"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Compute attention
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        return self.gamma * out + x


class AdaptiveFeatureExtractor(nn.Module):
    """Extract multi-scale features adaptively"""
    def __init__(self, channels):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(s) for s in [1, 2, 4]
        ])
        self.max_pool = nn.AdaptiveMaxPool2d(1)
    
    def forward(self, x):
        B = x.shape[0]
        features = []
        
        # Average pooling at multiple scales
        for pool in self.pools:
            features.append(pool(x).view(B, -1))
        
        # Max pooling for additional info
        features.append(self.max_pool(x).view(B, -1))
        
        return torch.cat(features, dim=1)


class RegressionHead(nn.Module):
    """Improved regression head with residual connections"""
    def __init__(self, input_dim, output_size, dropout=0.1):
        super().__init__()
        
        # Progressive dimension reduction
        dims = [input_dim, 256, 128, 64, 32]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        layers.append(nn.Linear(dims[-1], output_size))
        self.fc = nn.Sequential(*layers)
        
        # Skip connection from input to middle
        self.skip = nn.Linear(input_dim, 128)
    
    def forward(self, x):
        # Main path
        out = x
        for i, layer in enumerate(self.fc):
            out = layer(out)
            # Add skip connection at the middle layer
            if i == 4:  # After first 4 layers (at 128 dim)
                out = out + self.skip(x)
        
        return out


# ==============================
#   Alternative: Efficient Version
# ==============================

class EfficientUNetRegressor(nn.Module):
    """
    Lighter version with fewer parameters but similar performance.
    Good for limited compute resources.
    """
    def __init__(self, input_channels=1, output_size=5, base_channels=24, 
                 input_size=(64, 64), dropout_level=0.1, conv_block_type=0):
        super().__init__()
        

        if conv_block_type == 0:
            aConvBlock = ConvBlock

        # Smaller encoder (4 stages instead of 5)
        self.enc1 = aConvBlock(input_channels, base_channels, dropout_level)
        self.enc2 = aConvBlock(base_channels, base_channels * 2, dropout_level)
        self.enc3 = aConvBlock(base_channels * 2, base_channels * 4, dropout_level)
        self.enc4 = aConvBlock(base_channels * 4, base_channels * 8, dropout_level)
        self.enc5 = aConvBlock(base_channels * 8, base_channels * 16, dropout_level)
        
        self.pool = nn.AvgPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            aConvBlock(base_channels * 16, base_channels * 32, dropout_level),
            SEBlock(base_channels * 32)
        )
        
        # Decoder
        self.up5 = nn.ConvTranspose2d(base_channels * 32, base_channels * 16, 2, stride=2)
        self.dec5 = aConvBlock(base_channels * 32, base_channels * 16, dropout_level)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = aConvBlock(base_channels * 16, base_channels * 8, dropout_level)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = aConvBlock(base_channels * 8, base_channels * 4, dropout_level)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = aConvBlock(base_channels * 4, base_channels * 2, dropout_level)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = aConvBlock(base_channels * 2, base_channels, dropout_level)
        
        # Feature extraction
        self.feature_extractors = nn.ModuleList([
            AdaptiveFeatureExtractor(base_channels * 32),
            AdaptiveFeatureExtractor(base_channels * 16),
            AdaptiveFeatureExtractor(base_channels * 8),
            AdaptiveFeatureExtractor(base_channels * 4),
            AdaptiveFeatureExtractor(base_channels * 2),
            AdaptiveFeatureExtractor(base_channels)
        ])
        
        # Compute feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            features = self._forward_features(dummy)
            feature_dim = features.shape[1]
        
        # Simple regressor
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_level),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_level),
            nn.Linear(64, output_size)
        )
    
    def _forward_features(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc4(self.pool(e4))
        
        b = self.bottleneck(self.pool(e5))
        
        # Decoder with skip connections

        d5 = self.up4(b)
        d5 = self.dec4(torch.cat([d5, e5], dim=1))

        d4 = self.up4(d5)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Extract features
        features = [
            self.feature_extractors[0](b),
            self.feature_extractors[1](d5),
            self.feature_extractors[2](d4),
            self.feature_extractors[3](d3),
            self.feature_extractors[4](d2),
            self.feature_extractors[5](d1)
        ]
        
        return torch.cat(features, dim=1)
    
    def forward(self, x):
        features = self._forward_features(x)
        return self.regressor(features)