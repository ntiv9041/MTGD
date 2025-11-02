import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    # �� timesteps ת��Ϊ������������չΪ��ά���� Ȼ���� emb ��ˣ��㲥���ƻ�ʹÿ��ʱ�䲽���� emb �е�����Ԫ����ˡ�
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # ��� embedding_dim ��������ʹ�� torch.nn.functional.pad �ڵڶ���ά�������һ���㣬��ʹ���յ�Ƕ��ά���� embedding_dim һ�¡�
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ResnetBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
            self,
            channels,
            out_channels,
            emb_channels,
            dropout,
            use_conv=True,
            use_scale_shift_norm=True,
            use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            Normalize(channels),
            nn.SiLU(),
            torch.nn.Conv2d(channels,
                            self.out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                self.emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            Normalize(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            torch.nn.Conv2d(self.out_channels,
                            self.out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = torch.nn.Conv2d(channels,
                                                   self.out_channels,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1)
        else:
            self.skip_connection = torch.nn.Conv2d(channels,
                                                   self.out_channels,
                                                   kernel_size=1,
                                                   stride=1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class EncoderBranch(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.model_high = nn.Sequential(
            DoubleConv(1,64),
            Down(64, 128),
            Down(128, 128),
        )
        
        self.model_medium = Down(128, 256)
        self.model_low = Down(256, 256)

    def forward(self, x):
        o_high = self.model_high(x)
        o_medium= self.model_medium(o_high)
        o_low = self.model_low(o_medium)
        return [o_high,o_medium,o_low]

class AttnBlock(nn.Module):
    def __init__(self, in_channels, head=8):
        super().__init__()
        self.in_channels = in_channels
        self.head = head  # For 256ch, head=8 -> 32 per-head

        # Separate norms: critical to keep Q and KV channel stats independent
        self.norm_q = Normalize(in_channels)
        self.norm_kv = Normalize(in_channels)

        # QKV projections
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, kv):
        # Normalize independently
        q_in = self.norm_q(x)
        kv_in = self.norm_kv(kv)
    
        # Project to Q, K, V
        q = self.q(q_in)      # [B, C, Hq, Wq]
        k = self.k(kv_in)     # [B, C, Hk, Wk]
        v = self.v(kv_in)     # [B, C, Hk, Wk]
    
        b, c, h, w = q.shape
        hk, wk = k.shape[2], k.shape[3]
    
        # 🔧 NEW: align K/V spatial size to Q (critical fix)
        if (hk != h) or (wk != w):
            k = F.adaptive_avg_pool2d(k, (h, w))
            v = F.adaptive_avg_pool2d(v, (h, w))
    
        head = self.head if (c % self.head) == 0 else 1
        c_per_head = c // head
    
        # Optional one-time debug
        if not hasattr(self, "_dbg_spatial"):
            print(f"[DEBUG] MHA shapes after align: q={q.shape}, k={k.shape}, v={v.shape}, head={head}")
            self._dbg_spatial = True
    
        # Reshape for multi-head attention
        q = q.reshape(b * head, c_per_head, h * w).permute(0, 2, 1)  # (B*H, HW, C/H)
        k = k.reshape(b * head, c_per_head, h * w)                   # (B*H, C/H, HW)
        v = v.reshape(b * head, c_per_head, h * w)                   # (B*H, C/H, HW)
    
        # Scaled dot-product attention
        attn = torch.bmm(q, k) * (c_per_head ** -0.5)
        attn = torch.softmax(attn, dim=2)
    
        # Apply attention
        out = torch.bmm(v, attn.permute(0, 2, 1))  # (B*H, C/H, HW)
        out = out.reshape(b, c, h, w)
    
        # Output projection + residual
        out = self.proj_out(out)
        return x + out



class Encoder(nn.Module):
    def __init__(self,config,tm_ch):
        super().__init__()

        self.num_m = config.model.num_input_modality
        self.ch = config.model.ch


        self.Enc_list = nn.ModuleList()
        self.Res_list = nn.ModuleList()
        self.Gather_list = nn.ModuleList()
        self.DownC_list = nn.ModuleList()
        self.Atten_list = nn.ModuleList()
        for i in range(self.num_m):
            temp_ch = self.ch * (2 ** i)
            self.Enc_list.append(EncoderBranch())
            self.Res_list.append(ResnetBlock(temp_ch,temp_ch,tm_ch,False))
            self.Gather_list.append(DoubleConv(temp_ch * self.num_m,temp_ch))
            # determine base channel width for this stage
            attn_ch = 256
            self.Atten_list.append(AttnBlock(attn_ch))
            self.DownC_list.append(torch.nn.Conv2d(attn_ch * (self.num_m - 1),
                                                   attn_ch,
                                                   kernel_size=1,
                                                   stride=1,
                                                   padding=0))




    def forward(self, input,t_emb):

        encoder_output = []
        for i in range(len(self.Enc_list)):
            encoder_output.append(self.Enc_list[i](input[:,i]))

        E_result = [list(group) for group in zip(*encoder_output)]


        Att_in = [[x, [y for j, y in enumerate(E_result[-1]) if j != i]]
                  for i, x in enumerate(E_result[-1])]
        Att_out = []
        for i in range(len(self.Atten_list)):
            D_c = self.DownC_list[i]
            Att_out.append(self.Atten_list[i](Att_in[i][0],D_c(torch.cat(Att_in[i][1], dim=1))))

        E_result[-1] =  Att_out

        out_l = []
        for i in range(len(self.Res_list)):
            Ga_conv = self.Gather_list[i]
            ga = Ga_conv(torch.cat(E_result[i], dim=1))
            out_l.append(self.Res_list[i](ga, t_emb))

        return out_l

class Upsample(nn.Module):
    def __init__(self, in_channels,out_channels,with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Model(nn.Module):
    def __init__(self, config,n_channels=1, n_classes=1, bilinear=False):
        super(Model, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.ch = 64

        self.encoder = Encoder(config,self.ch)

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch//2,
                            self.ch//2),
            torch.nn.Linear(self.ch//2,
                            self.ch//2),
        ])
        
        self.pemb = nn.Module()
        self.pemb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch//2,
                            self.ch//2),
            torch.nn.Linear(self.ch//2,
                            self.ch//2),
        ])

        # Only project the encoder feature; x5 is already 512-ch
        self.q_proj = nn.Identity()
        self.kv_proj = nn.Identity()
        self.atten = AttnBlock(256)



        self.inc = DoubleConv(n_channels, self.ch)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.R1 = ResnetBlock(128,128,self.ch,dropout=False)
        self.down3 = Down(128, 256)
        self.R2 = ResnetBlock(256, 256, self.ch, dropout=False)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 256 // factor)
        self.up1 = Upsample(256,256)
        self.R3 = ResnetBlock(256, 256, self.ch, dropout=False)
        self.up2 = Upsample(256,128)
        self.R4 = ResnetBlock(128, 128, self.ch, dropout=False)
        self.up3 = Upsample(128,128)
        self.up4 = Upsample(128,64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x,t,condition,pet_type):
        temb = get_timestep_embedding(t, 32)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        
        
        pemb = get_timestep_embedding(pet_type, 32)
        pemb = self.pemb.dense[0](pemb)
        pemb = nonlinearity(pemb)
        pemb = self.pemb.dense[1](pemb)
        
        emb = torch.concat([temb,pemb],dim = 1)

        e_out = self.encoder(condition,emb)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.R1(x3,emb) + e_out[0]
        x4 = self.down3(x3)
        x4 = self.R2(x4,emb) + e_out[1]
        x5 = self.down4(x4)
        # Project both query and key/value to 512ch before attention
        x5 = self.q_proj(x5)
        kv = self.kv_proj(e_out[-1])
        x5 = self.atten(x5, kv)

        x = self.up1(x5)+x4
        x = self.R3(x,emb) + e_out[1]
        x = self.up2(x)+x3
        x = self.R4(x,emb) + e_out[0]
        x = self.up3(x)+x2
        x = self.up4(x)+x1
        logits = self.outc(x)
        return logits

















