# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numbers
from basicsr.models.archs.mamba import SSM

try:
    from thop import profile, clever_format
except ImportError:
    print("thop is not installed. FLOPs and parameter count will not be available.")
    profile = lambda model, inputs, verbose: (0, 0)
    clever_format = lambda x, y: (0, 0)


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = inputs[
            :, i, :, int(step * i) : int(step * i) + out_col
        ]
    return inputs[:, :, :, :out_col]


class InvertedResidual(nn.Module):
    def __init__(self, n_feats, expansion_factor=8):
        super().__init__()
        hidden_dim = n_feats * expansion_factor
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats, hidden_dim, 1, bias=False),
            nn.PReLU(),
            nn.Conv2d(
                hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False
            ),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, n_feats, 1, bias=False),
        )

    def forward(self, x):
        return x + self.conv(x)


class RLP_IR(nn.Module):
    def __init__(self, feat_c=40, depth=5, expansion=8):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, feat_c, 3, 1, 1, bias=True), nn.PReLU())
        self.blocks = nn.Sequential(
            *[InvertedResidual(feat_c, expansion) for _ in range(depth)]
        )
        self.head_mask = nn.Conv2d(feat_c, 1, 3, 1, 1, bias=True)
        self.head_feat = nn.Conv2d(feat_c, feat_c, 3, 1, 1, bias=True)

    def forward(self, img):
        feat = self.stem(img)
        feat = self.blocks(feat)
        rlp_map = self.head_mask(feat)
        rlp_feat = self.head_feat(feat)
        return rlp_map, rlp_feat


class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle,
            n_fea_middle,
            kernel_size=5,
            padding=2,
            bias=True,
            groups=n_fea_in,
        )

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


def replace_denormals(x, threshold=1e-10):
    y_real = x.real.clone()
    y_imag = x.imag.clone()
    y_real[(x.real < threshold) & (x.real > -1.0 * threshold)] = threshold
    y_imag[(x.imag < threshold) & (x.imag > -1.0 * threshold)] = threshold
    return torch.complex(y_real, y_imag)


class PGFA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, bias=False):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.internal_dim = dim_head * heads

        self.to_q = nn.Linear(dim, self.internal_dim, bias=bias)
        self.to_k = nn.Linear(dim, self.internal_dim, bias=bias)
        self.to_v = nn.Linear(dim, self.internal_dim, bias=bias)

        self.project_illu = nn.Conv2d(dim, self.internal_dim, kernel_size=1, bias=bias)
        self.project_rain = nn.Conv2d(dim, self.internal_dim, kernel_size=1, bias=bias)
        self.fusion = nn.Conv2d(
            self.internal_dim * 2, self.internal_dim, kernel_size=1, bias=bias
        )

        self.project_out = nn.Conv2d(self.internal_dim, dim, kernel_size=1, bias=True)

        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

        self.norm = LayerNorm(self.internal_dim, LayerNorm_type="WithBias")

    def forward(self, x_in, illu_fea_trans, rain_fea_trans):
        """
        x_in: [b,h,w,c]
        illu_fea_trans: [b,h,w,c]
        rain_fea_trans: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape

        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        q_img = (
            q_inp.reshape(b, h, w, self.internal_dim).permute(0, 3, 1, 2).contiguous()
        )
        k_img = (
            k_inp.reshape(b, h, w, self.internal_dim).permute(0, 3, 1, 2).contiguous()
        )
        v_img = (
            v_inp.reshape(b, h, w, self.internal_dim).permute(0, 3, 1, 2).contiguous()
        )

        illu_fea = illu_fea_trans.permute(0, 3, 1, 2).contiguous()
        rain_fea = rain_fea_trans.permute(0, 3, 1, 2).contiguous()
        rain_attn = self.project_rain(rain_fea)
        illu_attn = self.project_illu(illu_fea)
        prior = self.fusion(torch.cat([illu_attn, rain_attn], dim=1))

        v_guided = v_img * prior

        q_fft = torch.fft.rfft2(q_img.float())
        k_fft = torch.fft.rfft2(k_img.float())

        q_fft = replace_denormals(q_fft)
        k_fft = replace_denormals(k_fft)

        qk_fft = q_fft * k_fft
        qk_fft = replace_denormals(qk_fft)

        att_mag = torch.abs(qk_fft)
        q_phase = torch.angle(q_fft)
        k_phase = torch.angle(k_fft)
        att_phase = q_phase - k_phase

        real = att_mag * torch.cos(att_phase)
        imag = att_mag * torch.sin(att_phase)
        att_map_fft = torch.complex(real, imag)

        att_map = torch.fft.irfft2(att_map_fft, s=(h, w))

        att_map = self.norm(att_map)
        att_map_norm = att_map

        output = v_guided * att_map_norm

        out_c = self.project_out(output)

        out_p = self.pos_emb(out_c)

        out = out_c + out_p

        return out.permute(0, 2, 3, 1).contiguous()


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class DHNB(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([])

        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "attn": PGFA(dim=dim, dim_head=dim_head, heads=heads),
                        "ssm": SSM(hidden_dim=dim),
                        "film_gamma": nn.Conv2d(dim, dim, 1),
                        "film_beta": nn.Conv2d(dim, dim, 1),
                        "ff": PreNorm(dim, FeedForward(dim=dim)),
                    }
                )
            )

    def forward(self, x, illu_fea, rain_fea):
        x = x.permute(0, 2, 3, 1)

        for block in self.blocks:
            x = (
                block["attn"](
                    x, illu_fea.permute(0, 2, 3, 1), rain_fea.permute(0, 2, 3, 1)
                )
                + x
            )
            gamma = block["film_gamma"](rain_fea).permute(0, 2, 3, 1)
            beta = block["film_beta"](rain_fea).permute(0, 2, 3, 1)
            x_film = x * (1.0 + gamma) + beta
            x = block["ssm"](x_film) + x
            x = block["ff"](x) + x

        return x.permute(0, 3, 1, 2)


class Restorer(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=40, level=2, num_blocks=[1, 2, 2]):
        super(Restorer, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        DHNB(
                            dim=dim_level,
                            num_blocks=num_blocks[i],
                            dim_head=dim,
                            heads=dim_level // dim,
                        ),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                    ]
                )
            )
            dim_level *= 2

        # Bottleneck
        self.bottleneck = DHNB(
            dim=dim_level,
            dim_head=dim,
            heads=dim_level // dim,
            num_blocks=num_blocks[-1],
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            dim_level,
                            dim_level // 2,
                            stride=2,
                            kernel_size=2,
                            padding=0,
                            output_padding=0,
                        ),
                        nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                        DHNB(
                            dim=dim_level // 2,
                            num_blocks=num_blocks[level - 1 - i],
                            dim_head=dim,
                            heads=(dim_level // 2) // dim,
                        ),
                    ]
                )
            )
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea, rain_fea, rlp_map):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fu = torch.cat([x, rlp_map], dim=1)
        fea = self.embedding(fu)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        rain_fea_list = []
        for (
            DHNB,
            FeaDownSample,
            IlluFeaDownsample,
            RainFeaDownsample,
        ) in self.encoder_layers:
            fea = DHNB(fea, illu_fea, rain_fea)  # bchw
            illu_fea_list.append(illu_fea)
            rain_fea_list.append(rain_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)
            rain_fea = RainFeaDownsample(rain_fea)

        # Bottleneck
        fea = self.bottleneck(fea, illu_fea, rain_fea)

        # Decoder
        for i, (FeaUpSample, Fution, DHNB_Decoder) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            rain_fea = rain_fea_list[self.level - 1 - i]
            fea = DHNB_Decoder(fea, illu_fea, rain_fea)

        out = self.mapping(fea) + x

        return out


class DPDDN(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=3, n_feat=40, num_blocks=[1, 2, 2], level=2
    ):
        super(DPDDN, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.rlp_net = RLP_IR(feat_c=n_feat, depth=5, expansion=8)
        self.restorer = Restorer(
            in_dim=in_channels + 1,
            out_dim=out_channels,
            dim=n_feat,
            level=level,
            num_blocks=num_blocks,
        )

    def forward(self, img):
        B, C, H, W = img.shape
        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        rlp_map, rlp_feat = self.rlp_net(input_img)
        output_img = self.restorer(input_img, illu_fea, rlp_feat, rlp_map)

        return output_img


if __name__ == "__main__":
    # --- Configuration ---
    N_FEAT = 40
    NUM_BLOCKS = [1, 2, 2]  # Blocks per level in encoder/decoder
    LEVEL = 2
    INPUT_SIZE = (1, 3, 256, 256)
    import os

    # Check for GPU availability
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model Creation ---
    model = DPDDN(
        n_feat=N_FEAT,
        num_blocks=NUM_BLOCKS,
        level=LEVEL,
    ).to(device)

    # --- Model Analysis ---
    dummy_input = torch.randn(INPUT_SIZE).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print("=" * 70)
    print("DPDDN Model Analysis (Fused U-Net Architecture)")
    print(f"Input Tensor Size: {INPUT_SIZE}")
    print(f"Running on Device: {device}")
    print("-" * 70)
    print(f"Total Parameters: {params}")
    print(f"Total FLOPs: {flops}")
    print("=" * 70)

    # --- 测试时间大小256 ---
    print("\n" + "=" * 70)
    print("RDFormer 推理时间测试 (输入大小: 256x256)")
    print("=" * 70)

    # 预热GPU
    print("预热GPU...")
    for _ in range(10):
        _ = model(dummy_input)

    # 测试推理时间
    import time

    # 只测试一次
    print("进行单次推理测试...")

    with torch.no_grad():
        start_time = time.time()
        output = model(dummy_input)
        end_time = time.time()

        inference_time = end_time - start_time

    print(f"推理时间: {inference_time:.4f} 秒")
