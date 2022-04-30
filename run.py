from scripts.dataset.brixia.gen_data import gen_brixia_data
from scripts.dataset.chexpert.gen_data import gen_chexpert_data
from scripts.dataset.covid_ct.gen_data import gen_covid_ct_data
from scripts.dataset.covid_ct.parse_meta import parse_covid_ct_meta
from scripts.dataset.lidc.gen_data import gen_lidc_data
from scripts.dataset.lidc.parse_meta import parse_lidc_meta
from scripts.psp.generate import psp_generate
from scripts.psp.mix import psp_mix
from scripts.psp.train import psp_train
from scripts.stylegan.generate import stylegan_generate
from scripts.stylegan.train import stylegan_train
from utils.cli import OPTIONS
from utils.cli.dataset import DatasetGen, DatasetOptions, DatasetParse
from utils.cli.psp import PSPArch, PSPGenerate, PSPMix, PSPTrain
from utils.cli.stylegan import StyleGANArch, StyleGANGenerate, StyleGANTrain

if __name__ == "__main__":
    arch_options = OPTIONS.arch
    if isinstance(arch_options, StyleGANArch):
        if isinstance(arch_options.cmd, StyleGANTrain):
            stylegan_train()
        elif isinstance(arch_options.cmd, StyleGANGenerate):
            stylegan_generate()
        else:
            raise Exception("Invalid CLI arguments.")
    elif isinstance(arch_options, PSPArch):
        if isinstance(arch_options.cmd, PSPTrain):
            psp_train()
        elif isinstance(arch_options.cmd, PSPGenerate):
            psp_generate()
        elif isinstance(arch_options.cmd, PSPMix):
            psp_mix()
        else:
            raise Exception("Invalid CLI arguments.")
    elif isinstance(arch_options, DatasetOptions):
        if isinstance(arch_options.cmd, DatasetParse):
            if arch_options.dataset == "covid_ct":
                parse_covid_ct_meta()
            elif arch_options.dataset == "lidc":
                parse_lidc_meta()
            else:
                raise Exception("Invalid CLI arguments.")
        elif isinstance(arch_options.cmd, DatasetGen):
            if arch_options.dataset == "covid_ct":
                gen_covid_ct_data()
            elif arch_options.dataset == "lidc":
                gen_lidc_data()
            elif arch_options.dataset == "chexpert":
                gen_chexpert_data()
            elif arch_options.dataset == "brixia":
                gen_brixia_data()
            else:
                raise Exception("Invalid CLI arguments.")
        else:
            raise Exception("Invalid CLI arguments.")
    else:
        raise Exception("Invalid CLI arguments.")
