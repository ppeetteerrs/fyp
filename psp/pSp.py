from typing import Optional, Tuple

import torch
from stylegan.discriminator.discriminator import Discriminator
from stylegan.generator.generator import Generator
from torch import Tensor, nn
from utils.config import CONFIG

from psp.encoder import Encoder, EncoderDeep


class pSp(nn.Module):
    def __init__(self):
        super().__init__()

        # Define architecture
        if CONFIG.PSP_ENCODER == "original":
            self.encoder = Encoder(resolution=CONFIG.RESOLUTION).to("cuda")
        else:
            self.encoder = EncoderDeep(resolution=CONFIG.RESOLUTION).to("cuda")
        self.decoder = Generator.from_config().to("cuda")

        # Load model checkpoints
        ckpt = torch.load(str(CONFIG.PSP_CKPT))

        if CONFIG.PSP_LOSS_ID_DISCRIMINATOR > 0 or CONFIG.PSP_LOSS_DISCRIMINATOR > 0:
            self.discriminator = Discriminator.from_config().to("cuda")
        else:
            self.discriminator = None

        if "g_ema" in ckpt:
            self.decoder.load_state_dict(ckpt["g_ema"], strict=True)
            if self.discriminator is not None:
                self.discriminator.load_state_dict(ckpt["d"], strict=True)
            self.resumed = False
        else:
            self.encoder.load_state_dict(ckpt["encoder"], strict=True)
            self.decoder.load_state_dict(ckpt["decoder"], strict=True)
            if self.discriminator is not None:
                self.discriminator.load_state_dict(ckpt["discriminator"], strict=True)
            self.resumed = True

        # Load latent average
        self.latent_avg: Optional[Tensor]
        if CONFIG.PSP_USE_MEAN:
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
