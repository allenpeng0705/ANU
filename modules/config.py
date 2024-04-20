# The global config class for combining all the args / parameters together

import argparse
from modules.util import Singleton
from omegaconf import OmegaConf

@Singleton
class SDConfig(object):
    def __init__(self):
        super().__init__()
        self.cmd_opt = self.read_cmd_options()
        self.sd_config = OmegaConf.load(f"{self.cmd_opt.config}")

    def read_cmd_options(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--prompt",
            type=str,
            nargs="?",
            default="a painting of a virus monster playing guitar",
            help="the prompt to render"
        )
        parser.add_argument(
            "--work_mode",
            type=str,
            nargs="?",
            help="txt-to-img, img-to-img, inpaint",
            default="txt-to-img"
        )
        parser.add_argument(
            "--init_img",
            type=str,
            nargs="?",
            help="path to the input image, it's for img-to-img mode",
            default="demo.png"
        )
        parser.add_argument(
            "--strength",
            type=float,
            default=0.75,
            help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
        )
        parser.add_argument(
            "--outdir",
            type=str,
            nargs="?",
            help="dir to write results to",
            default="outputs/txt2img-samples"
        )
        parser.add_argument(
            "--outimgdir",
            type=str,
            nargs="?",
            help="dir to write results to",
            default="outputs/img2img-samples"
        )
        parser.add_argument(
            "--skip_grid",
            action='store_true',
            help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
        )
        parser.add_argument(
            "--skip_save",
            action='store_true',
            help="do not save individual samples. For speed measurements.",
        )
        parser.add_argument(
            "--ddim_steps",
            type=int,
            default=50,
            help="number of ddim sampling steps",
        )
        parser.add_argument(
            "--plms",
            action='store_true',
            help="use plms sampling",
        )
        parser.add_argument(
            "--dpm_solver",
            action='store_true',
            help="use dpm_solver sampling",
        )
        parser.add_argument(
            "--fixed_code",
            action='store_true',
            help="if enabled, uses the same starting code across samples ",
        )
        parser.add_argument(
            "--ddim_eta",
            type=float,
            default=0.0,
            help="ddim eta (eta=0.0 corresponds to deterministic sampling",
        )
        parser.add_argument(
            "--n_iter",
            type=int,
            default=2,
            help="sample this often",
        )
        parser.add_argument(
            "--H",
            type=int,
            default=512,
            help="image height, in pixel space",
        )
        parser.add_argument(
            "--W",
            type=int,
            default=512,
            help="image width, in pixel space",
        )
        parser.add_argument(
            "--C",
            type=int,
            default=4,
            help="latent channels",
        )
        parser.add_argument(
            "--f",
            type=int,
            default=8,
            help="downsampling factor",
        )
        parser.add_argument(
            "--n_samples",
            type=int,
            default=3,
            help="how many samples to produce for each given prompt. A.k.a. batch size",
        )
        parser.add_argument(
            "--n_rows",
            type=int,
            default=0,
            help="rows in the grid (default: n_samples)",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=7.5,
            help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        )
        parser.add_argument(
            "--from-file",
            type=str,
            help="if specified, load prompts from this file",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="configs/stable-diffusion/v1-inference.yaml",
            help="path to config which constructs model",
        )
        parser.add_argument(
            "--sdversion",
            type=str,
            default="v1",
            help="indicate the version of stable diffusion",
        )
        parser.add_argument(
            "--sdbase",
            type=str,
            default= "v1-5-pruned-emaonly.safetensors",
            #"V1-5-pruned.ckpt",
            help="name of the stable diffusion base model",
        )
        parser.add_argument(
            "--textencoder",
            type=str,
            default="openai/clip-vit-large-patch14",
            help="name of the text encoder model",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="the seed (for reproducible sampling)",
        )
        parser.add_argument(
            "--precision",
            type=str,
            help="evaluate at this precision",
            choices=["full", "autocast"],
            default="autocast"
        )

        return parser.parse_args()
