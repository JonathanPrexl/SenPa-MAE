import os
import hydra
from omegaconf import DictConfig, OmegaConf
from os.path import dirname as up

@hydra.main(version_base = None, config_path="./configs/pretrain", config_name="basemae")
def main(config : DictConfig) -> None:

    # we first have to set visible
    # devices before importing any torch libs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_idx)

    # check that the major configs are the same
    # for the current config file and the one
    # of the loaded checkpoint
    if config.resume:

        raise_error = False
        checkpoint_config_loc = os.path.join(up(up(config.resumeCheckpoint)),"used_parameters.json")

        checkpoint_config = OmegaConf.load(checkpoint_config_loc)
        checkpoint_config = OmegaConf.to_container(checkpoint_config,resolve=True)
        current_config = OmegaConf.to_container(config,resolve=True)

        skipkeys = ["gpu_idx","seed","nEpochs",
                    "outputpath","experimentname",
                    "resume","resumeCheckpoint",
                    "special_save_nEpoch",
                    "validate_after_every_n_epoch",
                    "plotting_every_N_sampels"]

        for key in current_config.keys():
            if not key in skipkeys:
                if not current_config[key] == checkpoint_config[key]:
                    raise_error = True
                    raise Warning(f"Configs of checkpoint does not match with current config for key {key}: \n \n {checkpoint_config[key]} \n \n {current_config[key]}")
        
        if raise_error:
            raise ValueError("configs dont match")

    # do fancy AI stuff
    trainer = hydra.utils.instantiate(config.trainroutine)
    trainer = trainer(config)

    trainer.fit()
    trainer.finalize()

if __name__  == "__main__":
    main()


