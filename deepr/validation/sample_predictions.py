import random
from pathlib import Path
from typing import Callable

import torch

from deepr.model.conditional_ddpm import cDDPMPipeline
from deepr.model.utils import get_hour_embedding
from deepr.utilities.logger import get_logger
from deepr.visualizations.giffs import generate_giff
from deepr.visualizations.plot_maps import plot_2_model_comparison
from deepr.visualizations.plot_samples import get_figure_model_samples

K_to_C = 273.15
logger = get_logger(__name__)


def sample_observation_vs_prediction(
    model,
    dataloader: torch.utils.data.DataLoader,
    local_dir: str,
    scaler_func: Callable = None,
    baseline: str = "bicubic",
    num_samples: int = 10,
    step: int = 1
):
    """
    Generate and save a comparison plot of model predictions and baseline samples.

    Parameters
    ----------
    model : object
        The neural network model used for predictions.
    dataloader : torch.utils.data.DataLoader
        The data loader used to fetch the data.
    local_dir : str
        The directory where the plot will be saved.
    scaler_func : Callable, optional
        A scaling function to apply on the data, by default None.
    baseline : str, optional
        The mode used for baseline interpolation, by default "bicubic".
    num_samples : int, optional
        The number of samples to randomly select and compare, by default 10.
    """
    samples_get = 0
    n_day = 0
    for elements in dataloader:
        try:
            era5, cerra, covars, times = elements
        except ValueError:
            era5, cerra, times = elements
            covars = None
        with torch.no_grad():
            pred_nn = model(era5, covariables=covars, return_dict=False)[0]
        samples_base = torch.nn.functional.interpolate(
            era5[..., 6:-6, 6:-6], scale_factor=5, mode=baseline
        )

        if scaler_func is not None:
            cerra = scaler_func(cerra, times[:, 2])
            samples_base = scaler_func(samples_base, times[:, 2])
            pred_nn = scaler_func(pred_nn, times[:, 2])

        for i in range(len(times)):
            n_day += 1
            if n_day % step != 0:
                continue
            # if random.choice([True, False]):
            #     continue
            filename = Path(local_dir) / f"pred_comparison_{samples_get}.png"
            t_str = f"{times[i, 0]:d}H {times[i, 1]:d}-{times[i, 2]:d}-{times[i, 3]:d}"
            plot_2_model_comparison(
                cerra[i, 0],
                samples_base[i, 0],
                pred_nn[i, 0],
                matrix_names=["CERRA", baseline.capitalize(), model.__class__.__name__],
                metric_name="ºC",
                date=t_str,
                filename=filename,
            )
            samples_get += 1
            if samples_get == num_samples:
                return None


def sample_diffusion_samples_random(
    pipeline: cDDPMPipeline,
    dataloader: torch.utils.data.DataLoader,
    scaler_func: Callable = None,
    baseline: str = "bicubic",
    num_samples: int = 10,
    num_realizations: int = 3,
    inference_steps: int = 1000,
    output_dir: str = None,
    device: str = "",
):
    n_samples = 0
    for i, (era5, cerra, times) in enumerate(dataloader):
        # Prepare data
        # 1 A) Encode hour
        hour_emb = get_hour_embedding(times[:, :1], "class", 24)

        # 1 B) Repeat each sample by number of realizations
        era5_repeated = era5.repeat(num_realizations, 1, 1, 1)

        if hour_emb is not None:
            hour_emb = hour_emb.to(device)
            hour_emb = hour_emb.repeat(num_realizations, 1).squeeze()

        # 1 C) Compute baseline predictions
        pred_base = torch.nn.functional.interpolate(
            era5[..., 6:-6, 6:-6], scale_factor=5, mode=baseline
        )

        # 2) Run the predictions
        pred_nn = pipeline(
            images=era5_repeated,
            class_labels=hour_emb,
            num_inference_steps=inference_steps,
            generator=torch.manual_seed(2023),
            output_type="tensor",
        ).images

        if scaler_func is not None:
            cerra = scaler_func(cerra, times[:, 2]) - K_to_C
            era5 = scaler_func(era5, times[:, 2]) - K_to_C
            pred_nn = (
                scaler_func(pred_nn, times[:, 2].repeat(num_realizations)) - K_to_C
            )
            pred_base = scaler_func(pred_base, times[:, 2]) - K_to_C

        # Make a grid out of the images
        sample_names = [f"{t[0]:d}H {t[1]:02d}-{t[2]:02d}-{t[3]:04d}" for t in times]
        get_figure_model_samples(
            cerra.cpu(),
            pred_nn.cpu(),
            input_image=era5.cpu(),
            baseline=pred_base.cpu(),
            column_names=sample_names,
            filename=output_dir + f"/samples_{i}.png",
        )
        n_samples += len(sample_names)
        if n_samples >= num_samples:
            return None


def sample_gif(
    pipeline,
    dataloader,
    scaler_func: Callable = None,
    output_dir: str = None,
    freq_timesteps_frame: int = 1,
    inference_steps: int = 1000,
    fps: int = 50,
    eta: float = 1,
):
    """
    Generate GIFs of the diffusion process for a given pipeline.

    Args:
    ----
        pipeline (callable): The pipeline function to apply to the images.
        dataloader (iterable): An iterable containing low-resolution reanalysis.
        scaler_func (callable, optional): A function to un-scale the images. Defaults to None.
        output_dir (str, optional): The directory to save the generated GIFs. Defaults to None.
        freq_timesteps_frame (int, optional): The frequency of diffusion timesteps to
            save as frames in the GIFs. Defaults to 1, which saves latents at all
            timesteps as frames.
        inference_steps (int, optional): The number of inference timesteps to perform
            the diffusion process. Defaults to 1000.
        fps (int, optional): The frames per second to show. Maximum value supported for
            most of modern browsers is 50fps.
    """
    era5, _, times = next(iter(dataloader))
    hr_im, interm = pipeline(
        images=era5,
        class_labels=times[:, :1],
        generator=torch.manual_seed(2023),
        eta=eta,
        num_inference_steps=inference_steps,
        return_dict=False,
        saving_freq_interm=freq_timesteps_frame,
        output_type="tensor",
    )
    for i, time in enumerate(times):
        date = f"{time[0]:d}H_{time[1]:02d}-{time[2]:02d}-{time[3]:04d}"
        logger.info(f"Generating GIF for time: {date}")
        fname = output_dir + f"/diffusion_{date}_{inference_steps}steps"
        generate_giff(interm[i], fname + "_scaled", fps=fps)

        scaled_interm = scaler_func(
            interm[i].unsqueeze(1), times[i, 2].repeat(interm.shape[1])
        )
        scaled_interm -= K_to_C
        generate_giff(scaled_interm.squeeze(), fname, label="Temperature (ºC)", fps=fps)


def diffusion_callback(
    model,
    scheduler,
    era5,
    cerra,
    times,
    scaler_func: Callable = None,
    output_dir: str = None,
    freq_timesteps_frame: int = 1,
    inference_steps: int = 1000,
    fps: int = 50,
    eta: float = 1,
    epoch: int = 0,
    **ddpm_kwargs,
):
    pipeline = cDDPMPipeline(unet=model, scheduler=scheduler, **ddpm_kwargs)
    pipeline.to(era5.device)
    hr_im, interm = pipeline(
        images=era5[:1],
        class_labels=times[:1, :1],
        generator=torch.manual_seed(2023),
        eta=eta,
        num_inference_steps=inference_steps,
        return_dict=False,
        saving_freq_interm=freq_timesteps_frame,
        output_type="tensor",
    )
    del era5
    times = times.cpu()
    date = f"{times[0, 0]:d}H_{times[0, 1]:02d}-{times[0, 2]:02d}-{times[0, 3]:04d}"
    logger.info(f"Generating GIFFs for time: {date}")
    fname = output_dir + f"/diffusion_{date}_{inference_steps}steps"
    if epoch is not None:
        fname += f"_{epoch}epoch"

    get_figure_model_samples(
        scaler_func(cerra[:1].cpu(), times[0, 2]),
        scaler_func(hr_im[:1].cpu(), times[0, 2]),
        filename=fname + "_comparison.png",
    )
    del cerra, hr_im

    # GIFFs
    interm = interm.cpu()
    generate_giff(interm[0], f"{fname}_scaled", fps=fps)

    interm = scaler_func(  # UNDO scaling of latents
        interm[0].unsqueeze(1), times[0, 2].repeat(interm.shape[1])
    )
    interm -= K_to_C
    generate_giff(interm.squeeze(), fname, label="Temperature (ºC)", fps=fps)
