import os
import logging
import wandb
import pathlib
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger


util_logger = logging.getLogger('meta_knowledge.utils.wandb_utils')

try:
    WANDB_CACHE = str(pathlib.PosixPath('~/.wandb_cache').expanduser())
except NotImplementedError:
    pathlib.PosixPath = pathlib.WindowsPath
    WANDB_CACHE = str(pathlib.PosixPath('~/.wandb_cache').expanduser())


def create_wandb_vars(config):
    """Creates special environment variables for trainers and other utilities 
    to use if such configuration values are provided

    :param config: the global configuration values 
    :raises: ValueError 
    """
    if config.wandb_name:
        os.environ["WANDB_NAME"] = config.wandb_name
    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project
    if config.wandb_entity:
        os.environ["WANDB_ENTITY"] = config.wandb_entity

    if config.wandb_name or config.wandb_project or config.wandb_entity:
        util_logger.info(
            'WANDB settings (options), name=%s, project=%s, entity=%s' %
            (config.wandb_name, config.wandb_project, config.wandb_entity))

def init_wandb_logger(config):
    """Initializes the wandb logger 

    :param config: the global configuration 
    """
    log_model = "all" if config.wandb_checkpoint else False
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_name,
        log_model=log_model
    )
    return wandb_logger


def download_wandb_data(config):
    """Downloads wandb data

    :param config: the global configuration 
    :rtype: None 
    """
    entity = config.wandb_entity if config.wandb_entity else ""
    project = "data-collection"
    alias = config.wandb_data
    dataset = config.dataset
    root = f"{config.data_dir}/{dataset}"

    api = wandb.Api()
    util_logger.info(f'Downloading data from wandb: {dataset}:{alias}')
    artifact = api.artifact(f'{entity}/{project}/{dataset}:{alias}', type='dataset')
    artifact_dir = artifact.download(root=root)
    util_logger.info('Dataset downloaded to %s' % artifact_dir)


def download_wandb_models(config):
    """Downloads any models as needed

    :param config: the global configuration 
    """
    model_name = config.checkpoint
    entity = config.wandb_entity if config.wandb_entity else ""
    project = config.wandb_project

    api = wandb.Api()
    util_logger.info(f'Downloading data from wandb: {model_name}:best_k')
    artifact = api.artifact(f"{entity}/{project}/{model_name}:best_k", type='model')
    artifact_dir = artifact.download(root=config.output_dir)
    checkpoints = [os.path.join(artifact_dir, f)
                   for f in os.listdir(artifact_dir) if '.ckpt' in f]
    if len(checkpoints) > 1:
        util_logger.warning('Multi-checkpoints found! Using first one...')
    
    config.checkpoint = os.path.abspath(checkpoints[0])

def setup_wandb(config):
    """Sets up wandb enviroment variables, downloads datasets, models, etc.. as needed

    :param config: the global configuration 
    :rtype: None 
    """
    download_wandb_data(config)

    if config.wandb_model:
        download_wandb_models(config)

    if config.wandb_project or config.wandb_entity:
        create_wandb_vars(config)    


class WandbArtifactCallback(pl.Callback):

    def __init__(
        self,
        output_dir,
        run_name,
        save_results=True,
        save_models=False,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.save_results = save_results
        self.save_models = save_models
        self.run_name = run_name

    def on_train_end(self, trainer, pl_module):
        # <-- add support for json format too
        dev_out = os.path.join(self.output_dir, "dev_eval.tsv")
        best_model_path = trainer.checkpoint_callback.best_model_path
        if trainer.global_rank > 0:
            return
        run = trainer.logger.experiment

        # if self.save_results and os.path.isfile(dev_out):
        #     file_artifact = wandb.Artifact("%s_results" % self.run_name, type="model_output")
        #     file_artifact.add_file(dev_out)
        #     run.log_artifact(file_artifact)

        if self.save_models and os.path.isfile(str(best_model_path)):
            model_artifact = wandb.Artifact(
                "%s_model" % self.run_name, type="model")
            model_artifact.add_file(best_model_path)
            run.log_artifact(model_artifact)

    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)


def save_wandb(config, metrics={}, runner=None):

    # first save metrics
    if config.save_wandb_results:

        # create new runner if didn't exist
        new_runner = False
        if runner is None:
            runner = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_name
            )
            new_runner = True

        # log metrics
        util_logger.info(f"Logging metrics to wandb: {metrics}...")
        runner.summary.update(metrics)

    # then log artifact output files
    dev_results = os.path.join(config.output_dir, f"{config.dev_name}_eval")
    train_results = os.path.join(
        config.output_dir, f"{config.train_name}_eval")
    test_results = os.path.join(config.output_dir, f"{config.test_name}_eval")
    dev_results += ".tsv" if not config.print_json else ".jsonl"
    train_results += ".tsv" if not config.print_json else ".jsonl"
    test_results += ".tsv" if not config.print_json else ".jsonl"

    util_logger.info('trying to save model output results')

    if config.save_wandb_results and (os.path.isfile(dev_results) or
                                      os.path.isfile(train_results) or os.path.isfile(test_results)):

        artifact = wandb.Artifact(
            '%s_out' % config.wandb_name.replace(">", "-"),
            type='model_output'
        )

        for oname, out_file in [
            ("dev", dev_results),
            ("train", train_results),
            ("test", test_results)
        ]:
            if not os.path.isfile(out_file):
                continue
            artifact.add_file(out_file)

        runner.log_artifact(artifact)

        if new_runner is True:
            runner.finish()
