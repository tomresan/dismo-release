import torch


dependencies = ["torch", "einops", "jaxtyping"]


def cogvideox5b_i2v(*, pretrained: bool = True, vae_slicing: bool = True, vae_tiling: bool = True, **kwargs):
    """
    Adapted CogVideoX-5B-I2V model.
    """
    from dismo.video_model_finetuning.cogvideox import CogVideoXMotionAdapter_5B_TI2V_Large

    model = CogVideoXMotionAdapter_5B_TI2V_Large(
        vae_slicing=vae_slicing,
        vae_tiling=vae_tiling,
        **kwargs
    )

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/CompVis/DisMo/resolve/main/cogvideox5b_i2v_large.pt",
            map_location="cpu",
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        assert len(unexpected) == 0
    
    model.requires_grad_(False)
    model.eval()
    return model


def motion_extractor_large(*, pretrained: bool = True, **kwargs):
    """
    DisMo's Motion Extractor.
    """
    from dismo.model import MotionExtractor_Large

    model = MotionExtractor_Large()

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/CompVis/DisMo/resolve/main/motion_extractor_large.pt",
            map_location="cpu",
        )
        model.load_state_dict(state_dict)
    
    model.requires_grad_(False)
    model.eval()
    return model