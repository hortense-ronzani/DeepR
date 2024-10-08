import os
from inspect import signature
from typing import Dict

import matplotlib.pyplot
import numpy as np
import torch
from accelerate import Accelerator, find_executable_batch_size
from accelerate.utils import LoggerType
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import Repository
from tqdm import tqdm

from deepr.data.generator import DataGenerator
from deepr.model.configs import TrainingConfig
from deepr.model.loss import compute_loss
from deepr.utilities.logger import get_logger
from deepr.visualizations.plot_maps import get_figure_model_samples
from deepr.visualizations.plot_loss import plot_loss

logger = get_logger(__name__)

model_kwargs = {"return_dict": False}


def save_samples(
    model,
    era5: torch.Tensor,
    cerra: torch.Tensor,
    num_epoch: int,
    output_name: str,
    **model_kwargs: Dict,
) -> matplotlib.pyplot.Figure:
    """
    Save a set of samples.

    Parameters
    ----------
    model : nn.Module
        The model used for generating samples.
    era5 : torch.Tensor
        The ERA5 data tensor.
    cerra : torch.Tensor
        The CERRA data tensor.
    output_name : str
        The output file name.

    Returns
    -------
    None
    """
    with torch.no_grad():
        images = model(era5, **model_kwargs)[0]

    pred_baseline = torch.nn.functional.interpolate(
        era5[..., 6:-6, 6:-6], scale_factor=5, mode="bicubic"
    )

    # Make a grid out of the images
    return get_figure_model_samples(
        cerra.cpu(),
        images.cpu(),
        input_image=era5.cpu(),
        baseline=pred_baseline.cpu(),
        num_epoch=num_epoch,
        filename=output_name,
        fig_size=(15, 10),
    )


def train_nn(
    config: TrainingConfig,
    model,
    train_dataset: DataGenerator,
    val_dataset: DataGenerator,
    hparams: Dict = {},
):
    """
    Train a neural network model.

    Parameters
    ----------
    config : TrainingConfig
        The training configuration.
    model : nn.Module
        The neural network model.
    train_dataset : DataGenerator
        The training dataset.
    val_dataset : DataGenerator
        The validation dataset.
    hparams : Dict, optional
        Hyperparameters.

    Returns
    -------
    model : nn.Module
        The trained model.
    repo_name : str
        The repository name.

    Notes
    -----
    This function performs the training of a neural network model using the provided
    datasets and configuration.
    """
    hparams = hparams | config.__dict__
    hparams.pop("__pydantic_initialised__", None)
    number_model_params = int(sum([np.prod(m.size()) for m in model.parameters()]))
    if "number_model_params" not in hparams:
        hparams["number_model_params"] = number_model_params

    model_name = model.__class__.__name__
    run_name = f"Train Super-Resolution NN ({model_name})"
    # aim_tracker = AimTracker(run_name, logging_dir="aim://10.9.64.88:31441")
    accelerator = Accelerator(
        cpu=config.device == "cpu",
        device_placement=True,
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=[LoggerType.TENSORBOARD],
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    
    # Load static covariables
    if False:
        if config.static_covariables is not None and len(config.static_covariables) > 0:
            hparams["static_covariables"] = ",".join(config.static_covariables)
            covars = []
            for static_covar in config.static_covariables:
                if static_covar in train_dataset.add_auxiliary_features.keys():
                    data = train_dataset.add_auxiliary_features[static_covar]
                    data = torch.from_numpy(data[list(data.data_vars)[0]].values)
                    if "orog" in static_covar:
                        data = (data - data.mean()) / data.std()
                    covars.append(data[np.newaxis, np.newaxis, ...])
                elif static_covar == "orog-diff":
                    orog_lr = train_dataset.add_auxiliary_features["orog-low"]["orog"]
                    orog_lr = torch.from_numpy(orog_lr.values)[np.newaxis, np.newaxis, ...]
                    orog_hr = train_dataset.add_auxiliary_features["orog-high"]["orog"]
                    orog_hr = torch.from_numpy(orog_hr.values)[np.newaxis, np.newaxis, ...]
                    pred_orog_hr = torch.nn.functional.interpolate(
                        orog_lr[..., 6:-6, 6:-6], scale_factor=5, modeq="bicubic"
                    )
                    diff = pred_orog_hr - pred_orog_hr
                    diff = (diff - diff.mean()) / diff.std()
                    covars.append(diff)
                else:
                    logger.info(f"Skipping covariable {static_covar}. Not recognized.")
            covars = torch.cat(covars, dim=1).to(config.device) if len(covars) > 0 else None
        else:
            covars = None

        if "covariables" in signature(model.forward).parameters.keys():
            model_kwargs["covariables"] = covars



    # operates with exponential decay, decreasing the batch size in half after each failed run
    #@find_executable_batch_size()
    def innner_training_loop(batch_size: int, model):
        
        nonlocal accelerator  # Ensure they can be used in our context
        accelerator.free_memory()  # Free all lingering references

        # Define important objects
        dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size, pin_memory=True
        )
        dataloader_val = torch.utils.data.DataLoader(
            val_dataset, batch_size, pin_memory=True
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(dataloader) * config.num_epochs),
        )

        if accelerator.is_main_process:
            if config.push_to_hub:
                repo = Repository(
                    config.output_dir,
                    clone_from=config.hf_repo_name,
                    token=os.getenv("HF_TOKEN"),
                )
                repo.git_pull()
            elif config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)
            if False:
                accelerator.init_trackers(run_name, config=hparams)
                tfboard_tracker = accelerator.get_tracker("tensorboard")

        (
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            model, optimizer, dataloader, dataloader_val, lr_scheduler
        )

        # Get fixed samples
        if config.static_covariables is not None and len(config.static_covariables) > 0:
            try:
                val_era5, val_cerra, val_covars = next(iter(val_dataloader))
                if batch_size > 4:
                    val_era5, val_cerra, val_covars = val_era5[:4], val_cerra[:4], val_covars[:4]
            except ValueError:
                val_era5, val_cerra = next(iter(val_dataloader))
            if batch_size > 4:
                val_era5, val_cerra = val_era5[:4], val_cerra[:4]
            val_covars = None
        else:
            val_era5, val_cerra = next(iter(val_dataloader))
            if batch_size > 4:
                val_era5, val_cerra = val_era5[:4], val_cerra[:4]
            val_covars = None

        # Initialization of loss saving
        loss_to_save = []
        l1_pred_to_save = []
        l1_lowres_to_save = []
        l1_blurred_to_save = []
        true_var_to_save = []
        bias_to_save = []
        mean_pred_to_save = []
        pred_var_to_save = []

        if False:
            tfboard_tracker.writer.add_graph(model, val_era5)
        logger.info(f"Number of parameters: {number_model_params}")
        global_step = 0
        # Now you train the model
        for epoch in range(config.num_epochs):
            progress_bar = tqdm(
                total=len(train_dataloader) + len(val_dataloader),
                disable=not accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch+1}")

            for elements in train_dataloader:
                try:
                    era5, cerra, covars = elements
                except:
                    era5, cerra = elements
                    covars = None
                # Predict the noise residual
                with accelerator.accumulate(model):
                    model_kwargs["covariables"] = covars
                    # model_kwargs["covariables"] has to be [batch_size:8, num_high_res_covars:2, 160, 240]
                    cerra_pred = model(era5, **model_kwargs)[0]
                    l1, l_lowres, l_blurred = compute_loss(cerra_pred, cerra)
                    if True:
                        loss = l1 + l_lowres + l_blurred
                    else:
                        loss = l1
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                cerra_pred_base = torch.nn.functional.interpolate(
                    era5[..., 6:-6, 6:-6], scale_factor=5, mode="bicubic"
                )

                l1_base, l_lowres_base, l_blurred_base = compute_loss(
                    cerra_pred_base, cerra
                )
                if True:
                    loss_base = l1_base + l_lowres_base + l_blurred_base
                else:
                    loss_base = l1_base
                progress_bar.update(1)
                pred_var = cerra_pred.var(keepdim=True, dim=0).mean().item()
                true_var = cerra.var(keepdim=True, dim=0).mean().item()
                lo = loss.detach().item()
                l_base = loss_base.detach().item()
                logs = {
                    "loss_vs_step": lo,
                    "l1_pred_vs_step": l1.mean().item(),
                    "l1_lowres_vs_step": l_lowres.mean().item(),
                    "l1_blurred_vs_step": l_blurred.mean().item(),
                    "baseline_loss_vs_step": l_base,
                    "improvement_vs_step": (l_base - lo) / l_base * 100,
                    "lr_vs_step": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                    "bias_perc_vs_step": (cerra - cerra_pred).mean().item()
                    / cerra.mean().item(),
                    "mean_var_ratio_vs_step": true_var / pred_var,
                    "epoch": epoch,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                if False:
                    tfboard_tracker.writer.add_histogram(
                        "cerra prediction", cerra_pred, global_step
                    )
                    tfboard_tracker.writer.add_histogram("cerra", cerra, global_step)
                global_step += 1

            # Evaluate
            loss, l1_pred, l1_lowres, l1_blurred = [], [], [], []
            true_var, pred_var, bias, mean_pred = [], [], [], []
            for elements in val_dataloader:
                try:
                    era5, cerra, covars = elements
                except:
                    era5, cerra = elements
                    covars = None
                # Predict the noise residual
                with torch.no_grad():
                    model_kwargs['covariables'] = covars
                    cerra_pred = model(era5, **model_kwargs)[0]
                    l_pred, l_lowres, l_blurred = compute_loss(cerra_pred, cerra)
                    loss.append((l_pred + l_lowres + l_blurred).mean().item())
                    l1_pred.append(l_pred.mean().item())
                    l1_lowres.append(l_lowres.mean().item())
                    l1_blurred.append(l_blurred.mean().item())

                pred_var.append(cerra_pred.var(keepdim=True, dim=0).mean().item())
                true_var.append(cerra.var(keepdim=True, dim=0).mean().item())
                bias.append((cerra - cerra_pred).mean().item())
                mean_pred.append(cerra_pred.mean().item())
                progress_bar.update(1)

            logs = {
                "val_loss_vs_epoch": sum(loss) / len(loss),
                "val_l1_vs_epoch": sum(l1_pred) / len(l1_pred),
                "val_l1_lowres_vs_epoch": sum(l1_lowres) / len(l1_lowres),
                "val_l1_blurred_vs_epoch": sum(l1_blurred) / len(l1_blurred),
                "val_bias_perc_vs_epoch": sum(bias) / sum(mean_pred),
                "val_mean_var_ratio_vs_epoch": sum(true_var) / sum(pred_var),
                "epoch": epoch,
            }
            accelerator.log(logs, step=epoch)
            progress_bar.close()

            # Save loss functions for each epoch
            loss_to_save.append(sum(loss) / len(loss))
            l1_pred_to_save.append(sum(l1_pred) / len(l1_pred))
            l1_lowres_to_save.append(sum(l1_lowres) / len(l1_lowres))
            l1_blurred_to_save.append(sum(l1_blurred) / len(l1_blurred))
            bias_to_save.append(sum(bias) / len(bias))
            mean_pred_to_save.append(sum(mean_pred) / len(mean_pred))
            true_var_to_save.append(sum(true_var) / len(true_var))
            pred_var_to_save.append(sum(pred_var) / len(pred_var))

            # After each epoch you optionally sample some demo images
            if accelerator.is_main_process:
                is_last_epoch = epoch == config.num_epochs - 1

                if (epoch + 1) % config.save_image_epochs == 0 or is_last_epoch:
                    logger.info("Saving sample predictions...")
                    samples_dir = os.path.join(config.output_dir, "samples")
                    os.makedirs(samples_dir, exist_ok=True)
                    model_kwargs['covariables'] = val_covars
                    fig = save_samples(
                        accelerator.unwrap_model(model),
                        val_era5,
                        val_cerra,
                        epoch+1,
                        output_name=f"{samples_dir}/{model_name}_{epoch+1:04d}.png",
                        **model_kwargs,
                    )
                    # Update loss plot
                    loss_dir = os.path.join(config.output_dir, "loss_function")
                    os.makedirs(loss_dir, exist_ok=True)
                    plot_loss(loss_to_save, "loss", config.num_epochs, loss_dir)

                    if is_last_epoch:
                        if False:
                            tfboard_tracker.writer.add_figure(
                                "Predictions", fig, global_step=epoch
                            )

                if (epoch + 1) % config.save_model_epochs == 0 or is_last_epoch:
                    logger.info("Saving model weights...")
                    model.save_pretrained(config.output_dir)
                    if config.push_to_hub:
                        repo.push_to_hub(
                            commit_message=f"Epoch {epoch+1}", blocking=True
                        )

        # Save loss as a function of epoch number
        np.save(config.output_dir+'/loss.npy', np.array(loss_to_save))
        np.save(config.output_dir+'/l1_pred.npy', np.array(l1_pred_to_save))
        np.save(config.output_dir+'/l1_lowres.npy', np.array(l1_lowres_to_save))
        np.save(config.output_dir+'/l1_blurred.npy', np.array(l1_blurred_to_save))
        np.save(config.output_dir+'/bias.npy', np.array(bias_to_save))
        np.save(config.output_dir+'/mean_pred.npy', np.array(mean_pred))
        np.save(config.output_dir+'/true_var.npy', np.array(true_var_to_save))
        np.save(config.output_dir+'/pred_var.npy', np.array(pred_var_to_save))

        return model

    trained_model = innner_training_loop(config.batch_size, model)

    accelerator.end_training()

    return trained_model, config.hf_repo_name
