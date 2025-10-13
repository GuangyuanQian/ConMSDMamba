import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from torch import Tensor
import torch.nn.init as init
from pytorch_wavelets import DWT1D, IDWT1D
from mamba_ssm.modules.mamba_simple import Mamba
from MSMamba import MultiScaleBiMamba
from torchinfo import summary
from thop import profile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class PointwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class DepthwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class ResidualConnectionModule(nn.Module):
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)


class ResidualConnectionModule_mask(nn.Module):
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnectionModule_mask, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if hasattr(self.module, 'forward') and 'mask' in inspect.signature(self.module.forward).parameters:
            return (self.module(inputs, mask) * self.module_factor) + (inputs * self.input_factor)
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)


class FeedForwardModule(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.sequential(inputs)
        return out


class my_AdaptiveAvgPool1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask = None):
        if mask is None:
            return nn.functional.adaptive_avg_pool1d(x, 1)

        mask = mask.unsqueeze(1).to(x.dtype)

        sum_pooled = torch.sum(x * mask, dim=2, keepdim=True)

        valid_count = torch.sum(mask, dim=2, keepdim=True)
        valid_count = valid_count.clamp(min=1e-9)

        return sum_pooled / valid_count


class TFConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
            wavelet: str = 'db4',
            wavelet_levels: int = 1,
            wavelet_filter_method: str = 'simple_scale',
            wavelet_conv_kernel: int = 3
    ) -> None:
        super().__init__()
        if DWT1D is None or IDWT1D is None:
            raise ImportError("pytorch_wavelets library is required but not found.")

        assert (kernel_size - 1) % 2 == 0, "kernel_size should be an odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"
        assert wavelet_filter_method in ['simple_scale', 'per_coeff_scale', 'conv'], \
            f"Invalid wavelet_filter_method: {wavelet_filter_method}"

        self.in_channels = in_channels
        self.wavelet_levels = wavelet_levels
        self.wavelet_filter_method = wavelet_filter_method

        self.transpose_pre = Transpose(shape=(1, 2))
        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.glu = GLU(dim=1)

        self.depthwise_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.bn_conv = nn.BatchNorm1d(in_channels)
        self.swish_conv = Swish()

        self.pw_conv_2_time = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)

        self.dwt = DWT1D(wave=wavelet, J=wavelet_levels, mode='symmetric')
        self.idwt = IDWT1D(wave=wavelet, mode='symmetric')

        if wavelet_filter_method == 'simple_scale':
            num_coeffs = 1 + wavelet_levels
            self.wavelet_scales = nn.Parameter(torch.ones(num_coeffs, 1, 1))

        elif wavelet_filter_method == 'per_coeff_scale':
            num_coeffs = 1 + wavelet_levels
            self.wavelet_coeff_scalers = nn.ModuleList([
                nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)
                for _ in range(num_coeffs)
            ])

        elif wavelet_filter_method == 'conv':
            num_coeffs = 1 + wavelet_levels
            self.wavelet_coeff_convs = nn.ModuleList([
                nn.Conv1d(
                    in_channels,
                    in_channels,
                    kernel_size=wavelet_conv_kernel,
                    padding='same',
                    bias=False
                ) for _ in range(num_coeffs)
            ])

            self.wavelet_coeff_norms = nn.ModuleList([
                 nn.BatchNorm1d(in_channels) for _ in range(num_coeffs)
            ])

        self.dropout = nn.Dropout(p=dropout_p)
        self.transpose_post = Transpose(shape=(1, 2))

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.transpose_pre(inputs)

        x_expanded = self.pw_conv_1(x)
        x_glu = self.glu(x_expanded)

        x_conv = self.depthwise_conv(x_glu)
        x_conv = self.bn_conv(x_conv)
        x_conv = self.swish_conv(x_conv)
        x_conv_out = self.pw_conv_2_time(x_conv)

        cA, cDs = self.dwt(x_glu)

        filtered_cA = cA
        filtered_cDs = list(cDs)

        if self.wavelet_filter_method == 'simple_scale':
            filtered_cA = cA * self.wavelet_scales[0]
            for i in range(self.wavelet_levels):
                filtered_cDs[i] = cDs[i] * self.wavelet_scales[i+1]

        elif self.wavelet_filter_method == 'per_coeff_scale':
            filtered_cA = self.wavelet_coeff_scalers[0](cA)
            for i in range(self.wavelet_levels):
                filtered_cDs[i] = self.wavelet_coeff_scalers[i+1](cDs[i])

        elif self.wavelet_filter_method == 'conv':
            filtered_cA_conv = self.wavelet_coeff_convs[0](cA)
            filtered_cA = self.wavelet_coeff_norms[0](filtered_cA_conv)
            for i in range(self.wavelet_levels):
                 cD_conv = self.wavelet_coeff_convs[i+1](cDs[i])
                 filtered_cDs[i] = self.wavelet_coeff_norms[i+1](cD_conv)

        x_wavelet = self.idwt((filtered_cA, filtered_cDs))

        target_length = x_glu.shape[-1]
        current_length = x_wavelet.shape[-1]
        if current_length > target_length:
            x_wavelet = x_wavelet[..., :target_length]
        elif current_length < target_length:
            padding_size = target_length - current_length

            last_values = x_wavelet[..., -1:]
            padding = last_values.repeat(1, 1, padding_size)
            x_wavelet = torch.cat((x_wavelet, padding), dim=-1)

        output_combined = x_conv_out + x_wavelet

        output_combined = self.dropout(output_combined)
        outputs = self.transpose_post(output_combined)

        return outputs


class ConmsdBlock(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(ConmsdBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.ResidualConn_A = ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            )

        self.ResidualConn_B = ResidualConnectionModule_mask(
            module=MultiScaleBiMamba(
                d_model=encoder_dim,
                d_state=16,
                d_conv=4,
                expand=2,
                scales=[1, 2, 4],
                dropout=0.1
            ),
        )

        self.ResidualConn_C = ResidualConnectionModule(
            module=TFConvModule(
                in_channels=encoder_dim,
                kernel_size=conv_kernel_size,
                expansion_factor=conv_expansion_factor,
                dropout_p=conv_dropout_p,
                wavelet='db4',
                wavelet_levels=2,
                wavelet_filter_method='per_coeff_scale'
            ),
        )

        self.ResidualConn_D = ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            )

        self.norm = nn.LayerNorm(encoder_dim)

        self.pooling = my_AdaptiveAvgPool1d()
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoder_dim // 2, 4)
        )

    def forward(self, inputs: Tensor, mask=None) -> Tensor:
        x1 = self.ResidualConn_A(inputs)
        x2 = self.ResidualConn_B(x1, mask)
        x3 = self.ResidualConn_C(x2)
        x4 = self.ResidualConn_D(x3)
        out = self.norm(x4)

        out = out.transpose(1, 2)
        out = self.pooling(out, mask)
        out = out.squeeze(-1)
        out = self.classifier(out)

        return out
