import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    @torch.inference_mode()
    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum('i,j -> ij', t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    @torch.inference_mode()
    def forward(self, x):
        cos, sin = self.cos[None, :x.size(-3), None, :], self.sin[None, :x.size(-3), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(self.head_dim)
        self.c_proj = CastedLinear(dim, dim)
        self.register_buffer('kv_cache', None, persistent=False)

    @torch.inference_mode()
    def forward(self, x, ve):
        B, T = x.size(0), x.size(1)
        
        # Generate Q, K, V
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim)
        
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
            
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        
        # Use KV cache if available
        if self.kv_cache is not None:
            k = torch.cat([self.kv_cache[0], k], dim=1)
            v = torch.cat([self.kv_cache[1], v], dim=1)
            self.kv_cache = torch.stack([k, v])

        # Efficient attention with flash attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2),  # (B, num_heads, T, head_dim)
                k.transpose(1, 2),  # (B, num_heads, T, head_dim)
                v.transpose(1, 2),  # (B, num_heads, T, head_dim)
                is_causal=True
            )
        else:
            # Fallback to regular attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(
                torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool(),
                float('-inf')
            )
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_()

    @torch.inference_mode()
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, model_dim, num_heads, use_attn=True):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads) if use_attn else None
        self.mlp = MLP(model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    @torch.inference_mode()
    def forward(self, x, ve, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve)
        x = x + self.mlp(norm(x))
        return x

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super().__init__()
        self.embed = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])

    @torch.inference_mode()
    def forward(self, inputs):
        ve = [emb(inputs).bfloat16() for emb in self.embed]
        ve = [ve[0], ve[1], ve[2], None, None, None, None, None, None, ve[0], ve[1], ve[2]]
        return ve

class ChronoGPT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, vocab_size, num_layers, num_heads, model_dim, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = vocab_size  # Store vocab_size as instance variable
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, use_attn=(i != 7))
                                   for i in range(num_layers)])
        self.value_embeds = ValueEmbedding(vocab_size, model_dim)
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.lm_head.weight.data.zero_()
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
    @torch.inference_mode()
    def forward(self, inputs, past_key_values=None):
        B = inputs.size(0) 
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)  # Add batch dimension if not present
        
        x0 = norm(self.embed(inputs).bfloat16())
        x = x0
        
        # Modify value embedding handling for batched input
        ve = [self.value_embeds(inputs[i].view(-1)) for i in range(B)]
        ve = [torch.stack([ve[b][i] for b in range(B)]) if ve[0][i] is not None else None 
              for i in range(len(ve[0]))]
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]

        # Handle cached states for batched input
        if past_key_values is not None:
            for i, block in enumerate(self.blocks):
                if block.attn is not None:
                    block.attn.kv_cache = past_key_values[i]

        present = []
        layer_outputs = []
        skip_connections = []

        # Process through encoder layers
        for i in range(self.num_encoder_layers):
            block = self.blocks[i]
            x = block(x, ve_enc[i], x0)
            if block.attn is not None:
                present.append(block.attn.kv_cache)
                block.attn.kv_cache = None
            skip_connections.append(x)
            layer_outputs.append(norm(x))

        # Process through decoder layers
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            block = self.blocks[self.num_encoder_layers + i]
            x = block(x, ve_dec[i], x0)
            layer_outputs.append(norm(x))
            if block.attn is not None:
                present.append(block.attn.kv_cache)
                block.attn.kv_cache = None

        x = norm(x)
        logits = self.lm_head(x)
        logits = 15 * torch.tanh(logits / 15)

        return logits.float(), layer_outputs
    @classmethod
    def from_pretrained(cls, repo_id, cache_dir=None, **kwargs):
        config_path = hf_hub_download(repo_id=repo_id, filename="config.pt", cache_dir=cache_dir)
        bin_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", cache_dir=cache_dir)
        config = torch.load(config_path)
        model = cls(**config)
        model.load_state_dict(torch.load(bin_path))
        return model

class ValueEmbedding_xl(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers=52):
        super().__init__()
        self.num_layers = num_layers
        # We only have 3 distinct embedding modules, reused at beginning and end.
        self.embed = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])

    def forward(self, inputs):
        # Compute the base embeddings (a list of length 3)
        base = [emb(inputs).bfloat16() for emb in self.embed]
        L = self.num_layers
        half = L // 2  # number of encoder layers (assumes num_layers is even)
        # Build encoder: first 3 layers get embeddings, rest get None.
        encoder = [base[i] if i < 3 else None for i in range(half)]
        # Build decoder: last 3 layers get embeddings, others get None.
        # For decoder layers, if i is in [half-3, half-1] then assign base[0], base[1], base[2]
        decoder = [base[i - (half - 3)] if i >= (half - 3) else None for i in range(half)]
        return encoder + decoder


class ChronoGPT_xl(nn.Module, PyTorchModelHubMixin):
    def __init__(self, vocab_size, num_layers, num_heads, model_dim, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = vocab_size  # Store vocab_size as instance variable
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, use_attn=True) for i in range(num_layers)])
        self.value_embeds = ValueEmbedding_xl(vocab_size, model_dim, num_layers=num_layers)
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.lm_head.weight.data.zero_()
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
    @torch.inference_mode()
    def forward(self, inputs, past_key_values=None):
        # Remove fixed batch size assumption
        B = inputs.size(0)  # Get batch size from input tensor
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)  # Add batch dimension if not present
        
        x0 = norm(self.embed(inputs).bfloat16())
        x = x0
        
        # Modify value embedding handling for batched input
        ve = [self.value_embeds(inputs[i].view(-1)) for i in range(B)]
        ve = [torch.stack([ve[b][i] for b in range(B)]) if ve[0][i] is not None else None 
              for i in range(len(ve[0]))]
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]

        # Handle cached states for batched input
        if past_key_values is not None:
            for i, block in enumerate(self.blocks):
                if block.attn is not None:
                    block.attn.kv_cache = past_key_values[i]

        present = []
        layer_outputs = []
        skip_connections = []

        # Process through encoder layers
        for i in range(self.num_encoder_layers):
            block = self.blocks[i]
            x = block(x, ve_enc[i], x0)
            if block.attn is not None:
                present.append(block.attn.kv_cache)
                block.attn.kv_cache = None
            skip_connections.append(x)
            layer_outputs.append(norm(x))

        # Process through decoder layers
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            block = self.blocks[self.num_encoder_layers + i]
            x = block(x, ve_dec[i], x0)
            layer_outputs.append(norm(x))
            if block.attn is not None:
                present.append(block.attn.kv_cache)
                block.attn.kv_cache = None

        x = norm(x)
        logits = self.lm_head(x)
        logits = 15 * torch.tanh(logits / 15)

        return logits.float(), layer_outputs
    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        config = {
            "model_type": "ChronoGPT",
            "vocab_size": self.embed.num_embeddings,
            "num_layers": len(self.blocks),
            "num_heads": self.num_heads,
            "model_dim": self.embed.embedding_dim
        }
        torch.save(config, os.path.join(save_directory, "config.pt"))
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)
    @classmethod
    def from_pretrained(cls, repo_id, cache_dir=None, **kwargs):
        config_path = hf_hub_download(repo_id=repo_id, filename="config.pt", cache_dir=cache_dir)
        bin_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", cache_dir=cache_dir)
        config = torch.load(config_path)
        model = cls(**config)
        model.load_state_dict(torch.load(bin_path))
        return model