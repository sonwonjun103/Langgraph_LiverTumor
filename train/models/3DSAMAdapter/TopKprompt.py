import torch
import math
import torch.nn as nn
import torch.nn.functional as F

"""
input의 liver를 prompt로 사용 => 이 친구에서 위치 tumor의 위치 파악?
transformer block 의 feature map 크기는 (1, 32, 32, 32, 256)
"""

class HeatMap3D(nn.Module):
    def __init__(self,
                 in_ch = 3,
                 base_ch = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_ch),
            nn.ReLU(inplace=True),

            nn.Conv3d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_ch * 2),
            nn.ReLU(inplace=True),

            nn.Conv3d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_ch * 4),
            nn.ReLU(inplace=True),

            nn.Conv3d(base_ch * 4, base_ch * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_ch * 4),
            nn.ReLU(inplace=True),

            nn.Conv3d(base_ch * 4, 1, kernel_size=1)
        )

    def forward(self, x):
        heat_logits = self.net(x)
        return heat_logits # (1, 1, 32, 32, 32)
    
class CoordPosMLP(nn.Module):
    def __init__(self,
                 out_dim = 768,
                 hidden = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, coords_num):
        return self.mlp(coords_num)
    
class TopKPromptTokenizer3D(nn.Module):
    def __init__(self,
                 k= 16,
                 add_pos=True,
                 token_dim = 768):
        super().__init__()

        self.k = k
        self.add_pos = add_pos
        self.pos_mlp = CoordPosMLP(out_dim = token_dim) if add_pos else None

    @staticmethod
    def _unravel_index(idx, D, H, W):
        w = idx % W
        idx2 = idx // W
        h = idx2 % H
        d = idx2 // H

        return d, h, w
    
    def forward(self, E, heat_logits):
        E = E.permute(0, 4, 1, 2, 3)
        B, C, D, H, W = E.shape
        heat = torch.sigmoid(heat_logits)
        heat_flat = heat.view(B, -1)

        topk_scores, topk_idx = torch.topk(heat_flat, k= self.k, dim = 1)
        #print(topk_idx, topk_scores)
        d, h, w = self._unravel_index(topk_idx, D, H, W)
        topk_coords = torch.stack([d, h, w], dim=-1)

        E_flat = E.view(B, C, -1).transpose(1, 2)
        idx_exp = topk_idx.unsqueeze(-1).expand(B, self.k, C)
        tokens = torch.gather(E_flat, dim=1, index=idx_exp)

        if self.add_pos:
            coords_norm = topk_coords.to(tokens.dtype)
            coords_norm[..., 0] = coords_norm[..., 0] / (D - 1) * 2 - 1
            coords_norm[..., 1] = coords_norm[..., 1] / (H - 1) * 2 - 1
            coords_norm[..., 2] = coords_norm[..., 2] / (W - 1) * 2 - 1
            #print(tokens.shape, self.pos_mlp(coords_norm).shape) # (1, 16, 3) (1, 16, 768)
            #tokens = tokens + self.pos_mlp(coords_norm) ## ?
            tokens = self.pos_mlp(coords_norm)

        return tokens, topk_scores, topk_coords
    
class LiverEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.heatnet1 = HeatMap3D(in_ch = 1, base_ch=32)
        self.heatnet2 = HeatMap3D(in_ch = 1, base_ch=32)
        self.heatnet3 = HeatMap3D(in_ch = 1, base_ch=32)

        self.fuse = nn.Conv3d(3, 1, kernel_size=1)

    def forward(self, x):
        heat_logits1 = self.heatnet1(x[:, 0, :, :, :].unsqueeze(1)) 
        heat_logits2 = self.heatnet2(x[:, 1, :, :, :].unsqueeze(1)) 
        heat_logits3 = self.heatnet3(x[:, 2, :, :, :].unsqueeze(1))

        heat_logits = torch.stack([heat_logits1, heat_logits2, heat_logits3], dim = 1).squeeze(2)
        heat_logits = self.fuse(heat_logits)

        # self attention 추가해야함

        return heat_logits
        
class TopkPromptModule(nn.Module):
    def __init__(self, k = 16, add_pos=True, token_dim = 256):
        super().__init__()
        self.tokenizer = TopKPromptTokenizer3D(k = k, add_pos=add_pos, token_dim=token_dim)

    def attn_gating(self, features, coords, scores, alpha=1.0):
        B = features.shape[0]
        K = coords.shape[1]

        F_mod = features.clone()

        for b in range(B):
            for k in range(K):
                d, h, w = coords[b, k]
                s = scores[b, k]
                F_mod[b, d, h, w, :] *= (1.0 + alpha * s)

        return F_mod
    
    def cross_attention(self, features, tokens):
        features = features.permute(0, 2, 3, 4, 1)
        B, D, H, W, C = features.shape
        K = tokens.shape[1]
        N = D * H * W

        F = features.view(B, N, C)
        Q = F
        K = tokens
        V = tokens
        attn_logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(C)
        attn = torch.softmax(attn_logits, dim=-1)

        out = torch.matmul(attn, V)
        out = out.view(B, D, H, W, C)

        return out

    def forward(self, map, tf):
        tokens, scores, coords = self.tokenizer(tf, map)
        attn_features = self.attn_gating(tf, coords, scores, alpha=1.0)
        attn_features = self.cross_attention(attn_features, tokens)

        #return tokens, scores, coords, heat_logits, attn_features
        attn_features = attn_features.permute(0, 4, 1, 2, 3)
        return attn_features


# if __name__=='__main__':
#     sample = torch.randn(1, 3, 512, 512, 512) # liver volume을 resize 한 거
#     features = torch.randn(1, 32, 32, 32, 256)
#     model = PromptFromVolume3D()
#     tokens, scores, coords, heat_logits, attn_features = model(sample, features)
#     print(f"Tokens : {tokens.shape}") # (1, 16, 768)
#     print(f"Scores : {scores.shape}") # (1, 16)
#     print(f"Coords : {coords.shape}") # (1, 16, 3)
#     #print(f"heat maps : {heat_maps.shape}")
#     print(f"Heat logits : {heat_logits.shape}") # (1, 1, 32, 32, 32)
#     print(f"Attn features : {attn_features.shape}")
    
#     print(coords)
#     print(scores)
    