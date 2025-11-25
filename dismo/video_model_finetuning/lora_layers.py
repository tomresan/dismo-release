import warnings
from typing import Optional, Union, Any

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lora import LoraLayer, Linear
from peft.tuners.lora.layer import _ConvNd


class DataProvider:
    def __init__(self):
        self.data = None

    def set(self, **kwargs):
        self.data = {k: v for k, v in kwargs.items()}

    def get(self, key, default=None):
        assert self.data is not None, "Error: need to set data first"
        if key not in self.data:
            return default
        return self.data[key]

    def reset(self):
        self.data = None


# Adapted peft.tuners.lora.Linear
class ConditionalLinear(Linear):
    adapter_layer_names: tuple[str, ...] = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_emb_gamma", "lora_emb_beta")

    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        d_cond: int,
        data_provider: DataProvider,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        data_pattern: str = "b (t l) d",
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)
        LoraLayer.__init__(self, base_layer, **kwargs)
        assert not use_dora, "DoRA variant not supported for ConditionalLinear for now"

        # NOTE: It is *ESSENTIAL* to name these modules with the lora_ prefix such that PEFT (more specifically the LoraModel class) doesn't disable their gradients ....
        self.lora_emb_gamma = nn.ModuleDict({})
        self.lora_emb_beta = nn.ModuleDict({})
        self.data_provider = data_provider
        self.data_pattern = data_pattern

        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
            d_cond=d_cond,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, d_cond, use_rslora, n_transformations = 1, use_dora = False, use_qalora = False, lora_bias = False, qalora_group_size = 32, **kwargs):
        #TODO: original loradapter has a single network for gamma and beta, why we use two separate ones?
        self.lora_emb_gamma[adapter_name] = nn.Linear(d_cond, r * n_transformations, bias=False)
        self.lora_emb_beta[adapter_name] = nn.Linear(d_cond, r * n_transformations, bias=False)
        nn.init.zeros_(self.lora_emb_gamma[adapter_name].weight)
        nn.init.zeros_(self.lora_emb_beta[adapter_name].weight)
        super().update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, use_qalora, lora_bias, qalora_group_size, **kwargs)

    def resolve_lora_variant(self, *, use_dora: bool, **kwargs):
        # No variants allowed for now (because missing implementations)
        return None

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        x = input
        # print("tokens shape", x.shape)
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                lora_emb_gamma = self.lora_emb_gamma[active_adapter]
                lora_emb_beta = self.lora_emb_beta[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)
                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    # result = result + lora_B(lora_A(dropout(x))) * scaling
                    cond: torch.Tensor = self.data_provider.get("cond_lora")
                    # cond has [B, 7, 1024] shape. 
                    # print("cond shape", cond.shape)
                    start_idx = self.data_provider.get("start_idx", 0)
                    cond_scale = lora_emb_gamma(cond) + 1.0
                    cond_shift = lora_emb_beta(cond)
                    x = lora_A(dropout(x))

                    # print("x shape after Ax", x.shape)
                    x_text, x_img = x[:, :start_idx], x[:, start_idx:]
                    x_img = rearrange(x_img, f"{self.data_pattern} -> b t l d", b=cond.shape[0], t=cond.shape[1])
                    x_img = (cond_scale[:, :, None] * x_img) + cond_shift[:, :, None]
                    x_img = rearrange(x_img, f"b t l d -> {self.data_pattern}")
                    x = torch.cat([x_text, x_img], dim=1)
                    result = result + lora_B(x) * scaling
                else:
                    raise NotImplementedError()
                    # result = self.lora_variant[active_adapter].forward(
                    #     self,
                    #     active_adapter=active_adapter,
                    #     x=x,
                    #     result=result,
                    # )

            result = result.to(torch_result_dtype)

        return result
    
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError()

    def unmerge(self) -> None:
        raise NotImplementedError()


# Adapted peft.tuners.lora._ConditionalConvNd
class _ConditionalConvNd(_ConvNd):
    adapter_layer_names: tuple[str, ...] = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_emb_gamma", "lora_emb_beta")

    # Lora implemented in a conv(2,3)d layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        d_cond: int,
        data_provider: DataProvider,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)
        LoraLayer.__init__(self, base_layer, **kwargs)
        assert not use_dora, "DoRA variant not supported for ConditionalLinear for now"

        if base_layer.groups > 1:
            warnings.warn("LoRA adapter added to ConvNd layer with groups > 1. Merging is not supported.")

        if r % base_layer.groups != 0:
            raise ValueError(
                f"Targeting a {base_layer.__class__.__name__} with groups={base_layer.groups} and rank {r}. "
                "Currently, support is limited to conv layers where the rank is divisible by groups. "
                "Either choose a different rank or do not target this specific layer."
            )
        
        # NOTE: It is *ESSENTIAL* to name these modules with the lora_ prefix such that PEFT (more specifically the LoraModel class) doesn't disable their gradients ....
        self.lora_emb_gamma = nn.ModuleDict({})
        self.lora_emb_beta = nn.ModuleDict({})
        self.data_provider = data_provider

        self._active_adapter = adapter_name
        self._kernel_dim = base_layer.weight.dim()

        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
            d_cond=d_cond,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, d_cond, use_rslora, n_transformations = 1, use_dora = False, lora_bias = False, **kwargs):
        self.lora_emb_gamma[adapter_name] = nn.Linear(d_cond, r * n_transformations, bias=False)
        self.lora_emb_beta[adapter_name] = nn.Linear(d_cond, r * n_transformations, bias=False)
        nn.init.zeros_(self.lora_emb_gamma[adapter_name].weight)
        nn.init.zeros_(self.lora_emb_beta[adapter_name].weight)
        # adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, lora_bias
        super().update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, lora_bias, **kwargs)

    def resolve_lora_variant(self, *, use_dora: bool, **kwargs):
        # No variants allowed for now (because missing implementations)
        return None

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)

        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                lora_emb_gamma = self.lora_emb_gamma[active_adapter]
                lora_emb_beta = self.lora_emb_beta[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)

                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    # result = result + lora_B(lora_A(dropout(x))) * scaling
                    cond: torch.Tensor = self.data_provider.get("cond_lora")
                    cond_scale = lora_emb_gamma(cond)+ 1.0
                    cond_shift = lora_emb_beta(cond)
                    x = lora_A(dropout(x))
                    x = rearrange(x, "(b t) c h w -> b t c h w", t=cond.shape[1])
                    x = (cond_scale[:, :, :, None, None] * x) + cond_shift[:, :, :, None, None]
                    x = rearrange(x, "b t c h w -> (b t) c h w")
                    result = result + lora_B(x) * scaling
                else:
                    raise NotImplementedError()
                    # result = self.lora_variant[active_adapter].forward(
                    #     self,
                    #     active_adapter=active_adapter,
                    #     x=x,
                    #     result=result,
                    # )

            result = result.to(torch_result_dtype)
        return result

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError()

    def unmerge(self) -> None:
        raise NotImplementedError()


class ConditionalConv2d(_ConditionalConvNd):
    # Lora implemented in a conv2d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 4:
            raise ValueError(f"Conv2d layer kernel must have 4 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv2d

    def resolve_lora_variant(self, *, use_dora: bool, **kwargs):
        # No variants allowed for now (because missing implementations)
        return None


class ConditionalConv1d(_ConditionalConvNd):
    # Lora implemented in a conv1d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 3:
            raise ValueError(f"Conv1d layer kernel must have 3 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv1d

    def resolve_lora_variant(self, *, use_dora: bool, **kwargs):
        # No variants allowed for now (because missing implementations)
        return None


class ConditionalConv3d(_ConditionalConvNd):
    # Lora implemented in a conv3d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 5:
            raise ValueError(f"Conv3d layer kernel must have 5 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv3d

    def resolve_lora_variant(self, *, use_dora: bool, **kwargs):
        # No variants allowed for now (because missing implementations)
        return None


class SimpleLoraLinear(torch.nn.Module):
    def __init__(
        self,
        out_features: int,
        in_features: int,
        c_dim: int,
        rank: int | float,
        data_provider: DataProvider,
        alpha: float = 1.0,
        lora_scale: float = 1.0,
        broadcast_tokens: bool = True,
        depth: int | None = None,
        use_depth: bool = False,
        n_transformations: int = 1,
        with_conditioning: bool = True,
        base_bias: bool = True,
        lora_bias: bool = False,
        frozen_weights_dtype=torch.bfloat16,
        target_path=None,
        **kwargs,
    ):
        super().__init__()

        self.data_provider = data_provider
        self.lora_scale = lora_scale
        self.broadcast_tokens = broadcast_tokens
        self.depth = depth
        self.use_depth = use_depth
        self.n_transformations = n_transformations
        self.rank = rank
        self.target_path = target_path

        # original weight of the matrix
        self.W = nn.Linear(in_features, out_features, bias=base_bias).to(dtype=frozen_weights_dtype)
        for p in self.W.parameters():
            p.requires_grad_(False)

        if type(rank) == float:
            self.rank = int(in_features * self.rank)

        self.A = nn.Linear(in_features, self.rank, bias=False)
        self.B = nn.Linear(self.rank, out_features, bias=lora_bias)

        nn.init.zeros_(self.B.weight)
        if lora_bias:
            nn.init.zeros_(self.B.bias)
        nn.init.kaiming_normal_(self.A.weight, a=1)

        self.with_conditioning = with_conditioning
        if with_conditioning:
            self.emb_gamma = nn.Linear(c_dim, self.rank * n_transformations, bias=False)
            self.emb_beta = nn.Linear(c_dim, self.rank * n_transformations, bias=False)

        # self.__old_A_weights = self.A.weight.detach().clone()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        # diff = (self.A.weight.detach().cpu() - self.__old_A_weights).abs().sum()
        w_out = self.W(x)

        if self.lora_scale == 0.0:
            return w_out

        c: torch.Tensor = self.data_provider.get("cond_lora")
        if self.use_depth:
            assert self.depth is not None, "block depth has to be provided"
            c = c[self.depth]

        if self.with_conditioning:
            scale = self.emb_gamma(c) + 1.0
            shift = self.emb_beta(c)

            # we need to do that when we only get a single embedding vector
            # e.g pooled clip img embedding
            # out is [B, 1, rank]
            if self.broadcast_tokens:
                scale = scale.unsqueeze(2)
                shift = shift.unsqueeze(2)

            if self.n_transformations > 1:
                # out is [B, 1, trans, rank]
                scale = scale.reshape(-1, 1, self.n_transformations, self.rank)
                shift = shift.reshape(-1, 1, self.n_transformations, self.rank)

        a_out = self.A(x.to(self.A.weight.dtype))  # [B, N, D]
        if self.n_transformations > 1:
            a_out = a_out.unsqueeze(-2).expand(-1, -1, self.n_transformations, -1)  # [B, N, trans, rank]

        # reshape for temporal injection
        if self.with_conditioning:
            a_out = rearrange(a_out, "b (t l) d -> b t l d", t=c.shape[1])
            a_cond = scale * a_out
            a_cond = a_cond + shift
            a_cond = rearrange(a_cond, "b t l d -> b (t l) d")
        else:
            a_cond = a_out

        if self.n_transformations > 1:
            a_cond = a_cond.mean(dim=-2)

        b_out = self.B(a_cond)

        return w_out + b_out.to(dtype=self.W.weight.dtype) * self.lora_scale
