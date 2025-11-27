from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.io import read_image, write_video, write_png, read_video
from dismo.video_model_finetuning.cogvideox import CogVideoXMotionAdapter_5B_TI2V_Large


def run(motion_video_path, source_image_path, prompt=""):
    original_video = read_video(motion_video_path)[0][::3][:25]
    if original_video.shape[0] < 25:
        return
    motion_video = original_video.float().div(255).mul(2).sub(1)
    src_image = read_image(source_image_path).permute(1, 2, 0).float().div(255).mul(2).sub(1)

    ckpt = torch.load("/export/home/tressler/dismo_release_checkpoints/cogvideox_5b_ti2v_large_600k_40k_koala.pt", map_location='cpu')
    cog = CogVideoXMotionAdapter_5B_TI2V_Large(
        motion_extractor_ckpt_path="/export/home/tressler/dismo_release_checkpoints/motion_extractor_large_600k.pt",
        vae_slicing=True,
        vae_tiling=True,
    )
    missing, unexpected = cog.load_state_dict(ckpt, strict=False)
    assert len(unexpected) == 0
    cog.eval()
    cog.requires_grad_(False)
    cog.cuda()

    gen_videos = cog.sample(
        motion_videos=motion_video.unsqueeze(0).cuda(),
        images=[src_image.cuda()],
        prompts=[prompt],
        progress_bar=True,
        n_steps=100,
        # text_guidance_scale=10.0,
        generator=torch.Generator('cuda').manual_seed(15), # 13 is good
        # negative_prompt="arms, hands, worst quality, inconsistent motion, blurry, jittery, distorted, ad pop-up, news pop-up",
    ).cpu()

    video_grid = torch.cat([original_video] + list(gen_videos.unbind(0)), dim=-2)
    write_video('output/transfer.mp4', video_grid, fps=8, options={"crf": "18", "preset": "slow"})

# run(
#     'videos/dirtbike_short.mov',
#     'images/castle_above.jpg',
#     prompt="The video shows a beautiful castle that is being filmed by a drone from above. The drone is moving very quickly. It quickly zooms down towards the castle and then flies off to the right. Fast camera trajectories. High quality. NO visible arms or hands."
# )

# run(
#     'videos/ballet_short2.mov',
#     'images/robot_dog.png',
#     prompt="The video shows a robotic boston dynamics dog that is performing a ballet. It is walking on its toes, kneeling down, and jumping up again in a ery elegant and control fashion. high quality. realistic motion. dance."
# )

# run(
#     'videos/ballet_short2.mov',
#     'images/gta_man.png',
#     prompt="The video shows a man that is performing a ballet in a GTA-style painting. The man is dancing on its toes, going down, jumping up again and rotating mid-air in an elegant and controlled fashion. high quality. realistic motion. dance."
# )

run(
    'videos/ballet_short2.mov',
    'images/fortnite_man.png',
    prompt="The video shows a man in Fortnite style that is performing a ballet. The man is dancing on its toes, going down, jumping up again and rotating mid-air in an elegant and controlled fashion. high quality. realistic motion. dance."
)

print('Done')