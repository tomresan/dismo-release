import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.io import read_image, write_video, write_png, read_video
from dismo.video_model_finetuning.cogvideox import CogVideoXMotionAdapter_5B_TI2V_Large


def run(motion_video_path, source_image_path, prompt="", output_dir="output/"):
    original_video = read_video(motion_video_path)[0][:25]
    if original_video.shape[0] < 25:
        return
    motion_video = original_video.float().div(255).mul(2).sub(1)
    src_image = read_image(source_image_path).permute(1, 2, 0).float().div(255).mul(2).sub(1)

    ckpt = torch.load("/export/home/tressler/dismo_release_checkpoints/cogvideox_5b_ti2v_large_600k_40k_koala.pt", map_location='cpu')
    cog = CogVideoXMotionAdapter_5B_TI2V_Large(
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
        n_steps=200,
        text_guidance_scale=3.5,
        # text_guidance_scale=10.0,
        generator=torch.Generator('cuda').manual_seed(19), # 13 is good
        negative_prompt="irregular body parts, incorrect anatomy, worst quality, inconsistent motion, blurry, jittery, distorted, ad pop-up, news pop-up",
    ).cpu()

    video_grid = torch.cat(list(gen_videos.unbind(0)), dim=-2)
    write_video(
        Path(output_dir).joinpath(f'{Path(motion_video_path).stem}' + '__to__' + f'{Path(source_image_path).stem}.mp4'), 
        video_grid, 
        fps=8, 
        options={
            "crf": "0",          # Lossless quality (best possible)
            "preset": "veryslow" # Highest-quality compression (slow)
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run motion transfer")
    parser.add_argument("--motion_video", type=str, required=True, help="Path to the motion video")
    parser.add_argument("--source_image", type=str, required=True, help="Path to the source image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output_dir", type=str, default="output", help="Text prompt")
    args = parser.parse_args()
    run(args.motion_video, args.source_image, args.prompt, args.output_dir)


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

# run(
#     'driving_videos/head_turner.mp4',
#     'source_images/paper_dragon2.png',
#     prompt="A low-poly dragon made out of vibrant-colored paper patches that is turned its head around while talking, origami-style, the video contains sharp details, articulated motion, and realistic motion of the dragon's head"
# )

# run(
#     'driving_videos/hand_puppet_slow.mp4',
#     'source_images/paper_dragon1.png',
#     prompt="A low-poly dragon made out of vibrant-colored paper patches that is turned its head around while talking, origami-style"
# )

print('Done')