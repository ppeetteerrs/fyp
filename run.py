from utils.cli import OPTIONS
from utils.cli.dataset import DatasetGen, DatasetOptions, DatasetParse
from utils.cli.psp import PSPArch, PSPGenerate, PSPMix, PSPTrain
from utils.cli.stylegan import StyleGANArch, StyleGANGenerate, StyleGANTrain

if __name__ == "__main__":
    arch_options = OPTIONS.arch
    if isinstance(arch_options, StyleGANArch):
        if isinstance(arch_options.cmd, StyleGANTrain):
            from scripts.stylegan.train import stylegan_train

            stylegan_train()
        elif isinstance(arch_options.cmd, StyleGANGenerate):
            from scripts.stylegan.generate import stylegan_generate

            stylegan_generate()
        else:
            raise Exception("Invalid CLI arguments.")
    elif isinstance(arch_options, PSPArch):
        if isinstance(arch_options.cmd, PSPTrain):
            from scripts.psp.train import psp_train

            psp_train()
        elif isinstance(arch_options.cmd, PSPGenerate):
            from scripts.psp.generate import psp_generate

            psp_generate()
        elif isinstance(arch_options.cmd, PSPMix):
            from scripts.psp.mix import psp_mix

            psp_mix()
        else:
            raise Exception("Invalid CLI arguments.")
    elif isinstance(arch_options, DatasetOptions):
        if isinstance(arch_options.cmd, DatasetParse):
            if arch_options.dataset == "covid_ct":
                from scripts.dataset.covid_ct.parse_meta import parse_covid_ct_meta

                parse_covid_ct_meta()
            elif arch_options.dataset == "lidc":
                from scripts.dataset.lidc.parse_meta import parse_lidc_meta

                parse_lidc_meta()
            else:
                raise Exception("Invalid CLI arguments.")
        elif isinstance(arch_options.cmd, DatasetGen):
            if arch_options.dataset == "covid_ct":
                from scripts.dataset.covid_ct.gen_data import gen_covid_ct_data

                gen_covid_ct_data()
            elif arch_options.dataset == "lidc":
                from scripts.dataset.lidc.gen_data import gen_lidc_data

                gen_lidc_data()
            elif arch_options.dataset == "chexpert":
                from scripts.dataset.chexpert.gen_data import gen_chexpert_data

                gen_chexpert_data()
            elif arch_options.dataset == "brixia":
                from scripts.dataset.brixia.gen_data import gen_brixia_data

                gen_brixia_data()
            else:
                raise Exception("Invalid CLI arguments.")
        else:
            raise Exception("Invalid CLI arguments.")
    else:
        raise Exception("Invalid CLI arguments.")
