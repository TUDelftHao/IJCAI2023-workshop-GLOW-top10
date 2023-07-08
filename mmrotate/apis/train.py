# Copyright (c) OpenMMLab. All rights reserved.
# Copied from mmdet, only modified `get_root_logger`.
import torch
import os
# import warnings

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, HOOKS)
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmcv.utils import build_from_cfg
from mmrotate.utils import compat_cfg, find_latest_checkpoint, get_root_logger

def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # if just swa training is performed,
    # skip building the runner for the traditional training

    if not cfg.get('only_swa_training', False):
        # build runner
        optimizer = build_optimizer(model, cfg.optimizer)

        if 'runner' not in cfg:
            cfg.runner = {
                'type': 'EpochBasedRunner',
                'max_epochs': cfg.total_epochs
            }
            # warnings.warn(
            #     'config is now expected to have a `runner` section, '
            #     'please set `runner` in your config.', UserWarning)
        else:
            if 'total_epochs' in cfg:
                assert cfg.total_epochs == cfg.runner.max_epochs

        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

        # an ugly workaround to make .log and .log.json filenames the same
        runner.timestamp = timestamp

        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
        elif distributed and 'type' not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

        # register hooks
        runner.register_training_hooks(
            cfg.lr_config,
            optimizer_config,
            cfg.checkpoint_config,
            cfg.log_config,
            cfg.get('momentum_config', None),
            custom_hooks_config=cfg.get('custom_hooks', None))

        if distributed:
            if isinstance(runner, EpochBasedRunner):
                runner.register_hook(DistSamplerSeedHook())

        # register eval hooks
        if validate:
            val_dataloader_default_args = dict(
                samples_per_gpu=1,
                workers_per_gpu=2,
                dist=distributed,
                shuffle=False,
                persistent_workers=False)

            val_dataloader_args = {
                **val_dataloader_default_args,
                **cfg.data.get('val_dataloader', {})
            }

            # Support batch_size > 1 in validation
            if val_dataloader_args['samples_per_gpu'] > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.val.pipeline = replace_ImageToTensor(
                    cfg.data.val.pipeline)
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)

            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_hook = DistEvalHook if distributed else EvalHook
            # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
            # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
            runner.register_hook(
                eval_hook(val_dataloader, **eval_cfg), priority='LOW')

        resume_from = None
        if cfg.resume_from is None and cfg.get('auto_resume'):
            resume_from = find_latest_checkpoint(cfg.work_dir)
        if resume_from is not None:
            cfg.resume_from = resume_from

        if cfg.resume_from:
            runner.resume(cfg.resume_from)
        elif cfg.load_from:
            runner.load_checkpoint(cfg.load_from)
        runner.run(data_loaders, cfg.workflow)
    else:
        # if just swa training is performed, there should be a starting model
        assert cfg.swa_resume_from is not None or cfg.swa_load_from is not None
    
    # perform swa training
    # build swa training runner
    if not cfg.get('swa_training', False):
        return
    from mmrotate.core import SWAHook

    logger.info('Start SWA training')
    swa_optimizer = build_optimizer(model, cfg.swa_optimizer)
    swa_runner = build_runner(
        cfg.swa_runner,
        default_args=dict(
            model=model,
            optimizer=swa_optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    swa_runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        swa_optimizer_config = Fp16OptimizerHook(
            **cfg.swa_optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.swa_optimizer_config:
        swa_optimizer_config = OptimizerHook(**cfg.swa_optimizer_config)
    else:
        swa_optimizer_config = cfg.swa_optimizer_config

    # register hooks
    swa_runner.register_training_hooks(cfg.swa_lr_config, swa_optimizer_config,
                                       cfg.swa_checkpoint_config,
                                       cfg.log_config,
                                       cfg.get('momentum_config', None))
    if distributed:
        if isinstance(swa_runner, EpochBasedRunner):
            swa_runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=8,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        # eval_cfg['work_dir'] = runner.work_dir
        eval_hook = DistEvalHook if distributed else EvalHook
        swa_runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
        swa_eval = True
        swa_eval_hook = eval_hook(
            val_dataloader, save_best='mAP', **eval_cfg)
    else:
        swa_eval = False
        swa_eval_hook = None

    # register swa hook
    swa_hook = SWAHook(
        swa_eval=swa_eval,
        eval_hook=swa_eval_hook,
        swa_interval=cfg.swa_interval)
    swa_runner.register_hook(swa_hook, priority='LOW')

    # register user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            swa_runner.register_hook(hook, priority=priority)

    if cfg.swa_resume_from:
        swa_runner.resume(cfg.swa_resume_from)
    elif cfg.swa_load_from:
        # use the best pretrained model as the starting model for swa training
        if cfg.swa_load_from == 'best_bbox_mAP.pth':
            best_model_path = os.path.join(cfg.work_dir, cfg.swa_load_from)
            # avoid the best pretrained model being overwritten
            new_best_model_path = os.path.join(cfg.work_dir,
                                               'best_bbox_mAP_pretrained.pth')
            if swa_runner.rank == 0:
                import shutil
                assert os.path.exists(best_model_path)
                shutil.copy(
                    best_model_path,
                    new_best_model_path,
                    follow_symlinks=False)
            cfg.swa_load_from = best_model_path
        swa_runner.load_checkpoint(cfg.swa_load_from)

    swa_runner.run(data_loaders, cfg.workflow)
