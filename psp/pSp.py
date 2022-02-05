from typing import Optional, Tuple

import torch
from stylegan.discriminator.discriminator import Discriminator
from stylegan.generator.generator import Generator
from torch import Tensor, nn
from utils.config import config

from psp.encoder import Encoder, EncoderV2


class pSp(nn.Module):
    def __init__(self):
        super().__init__()

        # self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        if config.PSP_ENCODER == "v1":
            self.encoder = Encoder(resolution=config.RESOLUTION).to("cuda")
        else:
            self.encoder = EncoderV2(resolution=config.RESOLUTION).to("cuda")
        self.decoder = Generator.from_config(config).to("cuda")
        self.discriminator = Discriminator.from_config(config).to("cuda")
        # Load model checkpoints
        ckpt = torch.load(str(config.PSP_CKPT))
        if config.PSP_CKPT_TYPE == "stylegan":
            self.decoder.load_state_dict(ckpt["g_ema"], strict=True)
            self.discriminator.load_state_dict(ckpt["d"], strict=True)
        else:
            self.encoder.load_state_dict(ckpt["encoder"], strict=True)
            self.decoder.load_state_dict(ckpt["decoder"], strict=True)

        # Load latent average
        self.latent_avg: Optional[Tensor]
        if config.PSP_USE_LATENT:
            if "latent_avg" in ckpt:
                self.latent_avg = ckpt["latent_avg"]
            else:
                self.latent_avg = self.decoder.mean_latent(10000)
        else:
            self.latent_avg = None

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:

        codes = self.encoder(input)

        if self.latent_avg is not None:
            codes += self.latent_avg.repeat(codes.shape[0], 1, 1)

        # No latent mask or injection
        images, _ = self.decoder([codes], input_type="w_plus", noises=None)

        return images, codes
