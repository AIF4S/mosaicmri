import argparse
import yaml
import os
from glob import glob
from tqdm.autonotebook import tqdm
import numpy as np

from fastmri.losses import SSIMLoss
from fastmri.data.transforms import center_crop
from fastmri.evaluate import ssim, psnr

from tqdm.autonotebook import tqdm
from runstats import Statistics

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

from .utils import logging, natural_sort
from .utils import get_dataloader, setup

import warnings
import uuid
from datetime import date, timedelta
import json
from pathlib import Path
from glob import glob


def _tensor_to_2d_np(x: torch.Tensor):
    t = x.detach().cpu()
    while t.ndim > 2:
        t = t[0]
    return t.numpy()


def _build_wandb_run(model_config, output_dir):
    if not bool(model_config.get("use_wandb", False)):
        return None
    try:
        import wandb

        run_name = model_config.get("wandb_name", Path(output_dir).stem)
        run = wandb.init(
            project=model_config.get("wandb_project", "mskmri-baselines"),
            entity=model_config.get("wandb_entity", None),
            name=run_name,
            dir=output_dir,
            mode=model_config.get("wandb_mode", "online"),
            config=model_config,
        )
        return run
    except Exception as e:
        warnings.warn(f"Failed to initialize W&B. Continuing without W&B. Error: {e}")
        return None


def _run_validation_epoch(
    model,
    forward_fn,
    loss_fn,
    model_config,
    eval_dataset_path,
    eval_transform,
    device,
    num_workers,
    epoch,
    wandb_run=None,
    wandb_log_images=False,
    wandb_num_images=0,
):
    model.eval()
    dataloader = get_dataloader(
        eval_dataset_path,
        eval_transform,
        num_workers=num_workers,
        augment_data=False,
        batch_size=1,
        shuffle=False,
        load_sensitivity_maps=True,
    )

    val_loss_stats = Statistics()
    ssim_stats = Statistics()
    psnr_stats = Statistics()
    logged_images = 0
    use_amp = bool(model_config.get("use_amp", False)) and device.startswith("cuda")
    amp_dtype = model_config.get("amp_dtype", "float16")
    autocast_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16

    for sample in tqdm(dataloader):
        target = sample.target.unsqueeze(-3).to(device)
        maxval = torch.clamp(sample.max_value.to(device), min=1e-8)

        with torch.no_grad():
            with autocast(enabled=use_amp, dtype=autocast_dtype):
                output = forward_fn(model, sample, device)
        output = output.float()
        output = center_crop(output, target.shape[-2:])

        # VarNet-style validation: evaluate on full cropped image by default.
        # Optional masking is only applied when explicitly enabled and mask has support.
        use_mask = bool(model_config.get("val_use_output_mask", False))
        if use_mask:
            output_mask = sample.output_mask.to(output.device)
            output_mask = center_crop(output_mask, target.shape[-2:])
            if torch.any(output_mask):
                output_m = output * output_mask
                target_m = target * output_mask
            else:
                output_m = output
                target_m = target
        else:
            output_m = output
            target_m = target

        loss = loss_fn(output_m, target_m, maxval).item()
        val_loss_stats.push(loss)

        maxval_scalar = float(sample.max_value.item())
        if not np.isfinite(maxval_scalar) or maxval_scalar <= 0:
            maxval_scalar = float(torch.max(target_m).detach().cpu().item())
            maxval_scalar = max(maxval_scalar, 1e-8)
        output_np = output_m.squeeze(0).detach().cpu().numpy()
        target_np = target_m.squeeze(0).detach().cpu().numpy()
        if not np.isfinite(output_np).all():
            output_np = np.nan_to_num(output_np, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(target_np).all():
            target_np = np.nan_to_num(target_np, nan=0.0, posinf=0.0, neginf=0.0)

        ssim_score = float(ssim(target_np, output_np, maxval_scalar).item())
        psnr_score = float(psnr(target_np, output_np, maxval_scalar).item())
        if not np.isfinite(ssim_score):
            ssim_score = 1.0 if np.allclose(output_np, target_np) else 0.0
        if not np.isfinite(psnr_score):
            psnr_score = 99.0 if np.allclose(output_np, target_np) else 0.0
        ssim_stats.push(ssim_score)
        psnr_stats.push(psnr_score)

        if wandb_run is not None and wandb_log_images and logged_images < int(wandb_num_images):
            import wandb

            output_img = _tensor_to_2d_np(output_m)
            target_img = _tensor_to_2d_np(target_m)
            if hasattr(sample, "image"):
                input_img = _tensor_to_2d_np(sample.image)
            else:
                input_img = output_img

            fname = sample.fname[0] if isinstance(sample.fname, (list, tuple)) else str(sample.fname)
            slice_num = int(sample.slice_num[0].item()) if hasattr(sample.slice_num, "__len__") else int(sample.slice_num.item())

            wandb_run.log(
                {
                    f"val/images/input_{logged_images}": wandb.Image(input_img, caption=f"{fname} slice={slice_num}"),
                    f"val/images/output_{logged_images}": wandb.Image(output_img, caption=f"{fname} slice={slice_num}"),
                    f"val/images/target_{logged_images}": wandb.Image(target_img, caption=f"{fname} slice={slice_num}"),
                },
                step=epoch,
            )
            logged_images += 1

    model.train()
    return {
        "val_loss": val_loss_stats.mean(),
        "val_ssim": ssim_stats.mean(),
        "val_psnr": psnr_stats.mean(),
    }


def train(
    rank,
    master_port,
    world_size,
    model_config,
    train_config,
    train_dataset_base_path,
    output_dir,
    finetune,
    num_workers,
    num_checkpoints,
    resume,
    model_seed,
):  
    accel_factors = model_config['accel_factors']
    lr = model_config['lr']
    batch_size = model_config['batch_size']
    #num_epochs = train_config['num_epochs']
    #samples_seen = train_config['samples_seen']
    num_epochs = train_config.num_epochs
    samples_seen = train_config.samples_seen
    grad_accum_steps = int(model_config.get("grad_accum_steps", 1))
    if grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

    total_steps = max(1, samples_seen // (batch_size * grad_accum_steps))
    augment_data = False
    assert batch_size % world_size == 0, 'batch_size must be divisible by world_size'

    device = 'cuda:%d' % rank 
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = master_port

        # initialize the process group
        # Validation can run for >10 minutes on rank 0 while other ranks wait
        # at synchronization points; increase timeout to avoid false NCCL aborts.
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(hours=2),
        )
        batch_size = batch_size//world_size
        
    model, data_transform, forward_fn = setup('train', model_config, accel_factors, seed=model_seed)
    _, eval_transform, _ = setup('test', model_config, accel_factors, seed=model_seed)
    model = model.to(device)

    if finetune is not None:
        checkpoint = torch.load(finetune, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    loss_fn = SSIMLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, 
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.01,
        anneal_strategy='linear',
        cycle_momentum=False,
        base_momentum=0., 
        max_momentum=0.,
        div_factor = 25.,
        final_div_factor=1.,
    )

    eval_dataset_path = model_config.get("val_dataset_path", None)
    val_every_n_epochs = int(model_config.get("val_every_n_epochs", 1))
    val_num_workers = int(model_config.get("val_num_workers", max(1, min(num_workers, 4))))
    use_amp = bool(model_config.get("use_amp", False)) and device.startswith("cuda")
    amp_dtype = model_config.get("amp_dtype", "float16")
    autocast_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    scaler = GradScaler(enabled=use_amp and autocast_dtype == torch.float16)

    wandb_run = None
    if rank == 0:
        wandb_run = _build_wandb_run(model_config, output_dir)

    # Training
    if not resume:
        logging(output_dir, 'Start training')
        first_epoch = 1
    else:
        assert glob(os.path.join(output_dir, 'checkpoints/*.pt')), 'Resume training failed: No checkpoints saved.'
        logging(output_dir, 'Resume training.')
        checkpoint = natural_sort(glob(os.path.join(output_dir, 'checkpoints/*.pt')))[-1]
        checkpoint = torch.load(checkpoint, map_location=device)
        checkpoint_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        logging(output_dir, 'Model from epoch {:d} loaded.'.format(checkpoint_epoch))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging(output_dir, 'Optimizer loaded. Current learning rate: {:f}'.format(optimizer.param_groups[0]['lr']))
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging(output_dir, 'Scheduler loaded. Current learning rate: {:f}'.format(scheduler.get_last_lr()[0]))
        first_epoch = checkpoint_epoch+1
        assert first_epoch <= num_epochs, 'Cannot resume. Training is already finished.'
        num_epochs = num_epochs-checkpoint_epoch
        print('Resume training at epoch {:d}'.format(first_epoch))
        logging(output_dir, 'Resume training at epoch {:d}'.format(first_epoch))

    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    model.train()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    path_to_checkpoints = os.path.join(output_dir, 'checkpoints')
    if rank == 0:
        if not os.path.exists(path_to_checkpoints):
            os.makedirs(path_to_checkpoints)
            print(f"Created {path_to_checkpoints}")
        else:
            print(f"{path_to_checkpoints} already exist. Continue.")
        
    for epoch in tqdm(range(first_epoch, first_epoch+num_epochs)):
        dataset = train_config.epochs_setup[epoch]
        dataset_path = os.path.join(train_dataset_base_path, dataset)
        dataloader = get_dataloader(dataset_path, data_transform, num_workers, augment_data, batch_size=batch_size, rank=rank, world_size=world_size, load_sensitivity_maps=False)

        if world_size > 1:
            dataloader.batch_sampler.set_epoch(epoch)

        total_loss = Statistics()
        local_num_iters = len(dataloader)
        effective_num_iters = local_num_iters
        if world_size > 1:
            iters_t = torch.tensor([local_num_iters], device=device, dtype=torch.int64)
            min_t = iters_t.clone()
            max_t = iters_t.clone()
            dist.all_reduce(min_t, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_t, op=dist.ReduceOp.MAX)
            effective_num_iters = int(min_t.item())
            if rank == 0 and int(max_t.item()) != effective_num_iters:
                warnings.warn(
                    "Per-rank dataloader length mismatch detected "
                    f"(min={effective_num_iters}, max={int(max_t.item())}). "
                    "Truncating each rank to min length to keep DDP synchronized."
                )
        optimizer.zero_grad(set_to_none=True)

        for it, sample in enumerate(tqdm(dataloader, total=effective_num_iters), start=1):
            if it > effective_num_iters:
                break
            target = sample.target.unsqueeze(-3).to(device)
            maxval = sample.max_value.to(device)
            with autocast(enabled=use_amp, dtype=autocast_dtype):
                output = forward_fn(model, sample, device)
            output = output.float()
            output = center_crop(output, target.shape[-2:])
            raw_loss = loss_fn(output, target, maxval)

            finite_flag = torch.isfinite(raw_loss).to(dtype=torch.int32)
            if world_size > 1:
                # Keep all ranks consistent: skip this step globally if any rank is non-finite.
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            if finite_flag.item() == 0:
                if rank == 0:
                    warnings.warn("Non-finite loss encountered; skipping batch on all ranks.")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss = raw_loss / grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            do_step = (it % grad_accum_steps == 0) or (it == effective_num_iters)
            if do_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                try:
                    scheduler.step()
                except ValueError:
                    warnings.warn("Scheduler total steps reached.")

            total_loss.push(raw_loss.item())

        train_loss = total_loss.mean()

        # Keep DDP ranks in lockstep before rank-0-only checkpoint/validation.
        if world_size > 1:
            dist.barrier()

        if rank == 0:
            with open(os.path.join(output_dir,'train_loss.txt'), 'a') as f:
                f.write(str(train_loss)+'\n')
            if epoch % num_checkpoints == 0:
                checkpoint = { 
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()}
                torch.save(checkpoint, os.path.join(path_to_checkpoints, 'epoch_'+str(epoch)+'.pt'))

            cur_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
            log_payload = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "lr": float(cur_lr),
            }

            if eval_dataset_path is not None and (epoch % val_every_n_epochs == 0):
                # Validation must not run through the DDP wrapper on a single rank.
                # Use the local module to avoid mismatched DDP collectives.
                val_model = model.module if world_size > 1 else model
                val_metrics = _run_validation_epoch(
                    model=val_model,
                    forward_fn=forward_fn,
                    loss_fn=loss_fn,
                    model_config=model_config,
                    eval_dataset_path=eval_dataset_path,
                    eval_transform=eval_transform,
                    device=device,
                    num_workers=val_num_workers,
                    epoch=epoch,
                    wandb_run=wandb_run,
                    wandb_log_images=bool(model_config.get("wandb_log_images", False)),
                    wandb_num_images=int(model_config.get("wandb_num_images", 0)),
                )
                log_payload.update(val_metrics)
                print(
                    "Epoch {}, Train loss: {:0.4f}, Val loss: {:0.4f}, Val SSIM: {:0.4f}, Val PSNR: {:0.2f}".format(
                        epoch,
                        log_payload["train_loss"],
                        log_payload["val_loss"],
                        log_payload["val_ssim"],
                        log_payload["val_psnr"],
                    )
                )
            else:
                print("Epoch {}, Train loss.: {:0.4f}".format(epoch, log_payload["train_loss"]))

            if wandb_run is not None:
                wandb_run.log(log_payload, step=epoch)

        # Keep DDP ranks synchronized after rank-0-only validation/logging.
        if world_size > 1:
            dist.barrier()

    logging(output_dir, 'Finish training')
    if wandb_run is not None:
        wandb_run.finish()

    if world_size > 1:
        print('Destroying process group...')
        dist.destroy_process_group()
        print('Done.')

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_setup", 
        type=str,
        required=True,
        help="Configuration of the model setup as yaml file (model, acceleration factor, batch_size, learning_rate)"
    )
    parser.add_argument(
        "--train_setup", 
        type=str,
        required=True,
        help="Configuration of the training setup as yaml file (which dataset at which epoch)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        required=True,
        help="Path to directory where checkpoints and logs will be stored."
    )
    parser.add_argument(
        "--path_to_model_summary", 
        type=str,
        default = None,
        help="Name and path to output file as .json"
    )
    parser.add_argument(
        "--finetune", 
        type=str, 
        default=None,
        help="Path to checkpoint for finetuning"
    )
    parser.add_argument(
        "--num_checkpoints", 
        type=int, 
        default=5, # before: 5, but should be part of configuration anyways
        help="Saves a checkpoint after each 'num_checkpoints' epochs of training."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="Number of workers for pytorch dataloader"
    )
    parser.add_argument(
        "--resume", 
        action='store_true',
        default=False,
        help="Whether to resume training from last checkpoints contained in --output_dir"
    )
    parser.add_argument(
        "--world_size", 
        type=int, 
        default=1,
        help="Set larger than 1 for DDP."
    )
    parser.add_argument(
        "--model_seed", 
        type=int, 
        default=0,
        help="Random seed for init the model"
    )
    
    return parser
    

def main(args, model_config, train_config, train_dataset_base_path):
 
    train_args = {
        'world_size': args.world_size,
        'model_config': model_config,
        'train_config': train_config,
        'train_dataset_base_path' : train_dataset_base_path,
        'output_dir': args.output_dir,
        'finetune': args.finetune,
        'num_workers': args.num_workers,
        'num_checkpoints': args.num_checkpoints,
        'resume': args.resume,
        'model_seed': args.model_seed,
    }

    if train_args['world_size'] > 1:
        master_port = str(np.random.randint(15000,65535)); print(f"master port: {master_port}")
        mp.spawn(train, args=(master_port, )+tuple(train_args.values()), nprocs=train_args['world_size'], join=True)
    else:
        train(0, None, **train_args)

    exp_setup = dict({k:v for k,v in train_args.items() if k not in ['output_dir', 'resume', 'train_config']})
    exp_setup["train_config"] = train_config.dict()

    model_data = {
        "name": train_config.name,
        "uuid": str(uuid.uuid4()),
        "creation_date": str(date.today()),
        "output_dir": os.getcwd(),
        "checkpoints": dict((Path(f).stem, f) for f in natural_sort(glob(os.path.join(train_args['output_dir'], 'checkpoints', '*.pt')))),
        "exp_setup": exp_setup
    }
    model_data_json = json.dumps(model_data, indent=4)
    with open(args.path_to_model_summary, 'w') as outfile:
        outfile.write(model_data_json)

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    main(args)
    
