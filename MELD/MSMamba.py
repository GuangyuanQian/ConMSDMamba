import torch
from mamba_ssm.modules.mamba_simple import Mamba
from torch import nn
import torch.nn.functional as F
from torch import Tensor


class ExBimamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            device=None,
            dtype=None,
            Amatrix_type='default'
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.forward_mamba = Mamba(d_model=self.d_model, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
        self.backward_mamba = Mamba(d_model=self.d_model, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
        self.output_proj = nn.Linear(2 * self.d_model, self.d_model)

    def forward(self, hidden_input, mask=None):
        forward_output = self.forward_mamba(hidden_input)

        if mask is not None:
            lengths = mask.sum(dim=1).long()

            batch_size = hidden_input.size(0)
            backward_inputs = []
            for i in range(batch_size):
                valid_len = lengths[i]
                seq = hidden_input[i]

                valid_part = seq[:valid_len]
                padding_part = seq[valid_len:]

                flipped = torch.cat([
                    valid_part.flip(0),
                    padding_part
                ])
                backward_inputs.append(flipped)

            backward_input = torch.stack(backward_inputs)

            backward_output = self.backward_mamba(backward_input)

            backward_outputs = []
            for i in range(batch_size):
                valid_len = lengths[i]
                seq = backward_output[i]

                valid_part = seq[:valid_len]
                padding_part = seq[valid_len:]

                restored = torch.cat([
                    valid_part.flip(0),
                    padding_part
                ])
                backward_outputs.append(restored)

            backward_output = torch.stack(backward_outputs)
        else:
            backward_output = self.backward_mamba(hidden_input.flip([1]))
            backward_output = backward_output.flip([1])

        res = torch.cat((forward_output, backward_output), dim=-1)
        res = self.output_proj(res)

        return res


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1))
        max_out = self.mlp(self.max_pool(x).squeeze(-1))

        channel_att = torch.sigmoid(avg_out + max_out)
        out = x * channel_att.unsqueeze(-1)
        return out


class MultiScaleBiMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            scales=[1, 2, 4],
            dropout=0.1,
            device=None,
            dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.scales = scales
        self.num_scales = len(scales)

        self.d_scale = d_model // self.num_scales
        assert d_model % self.num_scales == 0, f"d_model ({d_model}) must be divisible by number of scales ({self.num_scales})"

        self.bimambas = nn.ModuleList([
            ExBimamba(
                d_model=self.d_scale,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                **factory_kwargs
            ) for _ in scales
        ])

        self.pools = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    self.d_scale,
                    self.d_scale,
                    kernel_size=3,
                    stride=1,
                    padding='same',
                    dilation=scale,
                    groups=self.d_scale
                ),
                nn.BatchNorm1d(self.d_scale),
                nn.GELU(),
            ) for scale in scales
        ])

        self.channel_attention = ChannelAttention(d_model)

        self.fusion = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden_input, mask=None):
        batch_size, seq_len, _ = hidden_input.shape

        xs = torch.chunk(hidden_input, self.num_scales, dim=-1)

        outs = []

        for i, (scale_x, bimamba, pool) in enumerate(zip(xs, self.bimambas, self.pools)):
            scale_x_conv = pool(scale_x.transpose(1, 2))
            scale_x_conv = scale_x_conv.transpose(1, 2)

            out = bimamba(scale_x_conv, mask)
            outs.append(out)

        output = torch.cat(outs, dim=-1)

        output = output.transpose(1, 2)
        output = self.channel_attention(output)
        output = output.transpose(1, 2)

        output = self.fusion(output)

        output = self.dropout(output)

        return output
