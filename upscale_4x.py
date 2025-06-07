#!/usr/bin/env python3
"""
Stable Diffusion 4x Upscaler

A production-ready CLI tool for upscaling images using Stable Diffusion x4 upscaler model
with memory-optimized inference pipeline.
"""

import gc
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import click
import torch
from dotenv import load_dotenv
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from diffusers.models.attention_processor import AttnProcessor2_0

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UpscalePipeline:
    """Memory-optimized Stable Diffusion 4x upscaler pipeline."""

    def __init__(self, model_path: str, dtype: Optional[torch.dtype] = None):
        """
        Initialize the upscale pipeline.

        Args:
            model_path: Path to the Stable Diffusion x4 upscaler model
            dtype: Data type for model weights (auto-detected from env if None)
        """
        self.model_path = model_path

        # Set dtype from environment or default
        if dtype is None:
            dtype_str = os.getenv("UPSCALE_DTYPE", "bfloat16").lower()
            if dtype_str == "float16":
                self.dtype = torch.float16
            elif dtype_str == "float32":
                self.dtype = torch.float32
            else:
                self.dtype = torch.bfloat16
        else:
            self.dtype = dtype

        self.pipeline = None

        # Load optimization settings from environment
        self.enable_attention_slicing = (
            os.getenv("UPSCALE_ENABLE_ATTENTION_SLICING", "true").lower() == "true"
        )
        self.enable_vae_tiling = (
            os.getenv("UPSCALE_ENABLE_VAE_TILING", "true").lower() == "true"
        )
        self.enable_cpu_offload = (
            os.getenv("UPSCALE_ENABLE_CPU_OFFLOAD", "true").lower() == "true"
        )
        self.enable_xformers = (
            os.getenv("UPSCALE_ENABLE_XFORMERS", "true").lower() == "true"
        )

    def load_pipeline(self) -> None:
        """Load and configure the upscaling pipeline with memory optimizations."""
        logger.info("Loading upscaling pipeline components...")

        with torch.amp.autocast("cuda", dtype=self.dtype):
            # Load components separately for better memory management
            logger.info("Loading tokenizer...")
            tokenizer = CLIPTokenizer.from_pretrained(
                self.model_path, subfolder="tokenizer", torch_dtype=self.dtype
            )

            logger.info("Loading text encoder...")
            text_encoder = CLIPTextModel.from_pretrained(
                self.model_path, subfolder="text_encoder", torch_dtype=self.dtype
            )

            logger.info("Loading VAE...")
            vae = AutoencoderKL.from_pretrained(
                self.model_path, subfolder="vae", torch_dtype=self.dtype
            )

            logger.info("Loading UNet...")
            unet = UNet2DConditionModel.from_pretrained(
                self.model_path, subfolder="unet", torch_dtype=self.dtype
            )

            logger.info("Loading schedulers...")
            scheduler = DDIMScheduler.from_pretrained(
                self.model_path, subfolder="scheduler"
            )
            low_res_scheduler = DDPMScheduler.from_pretrained(
                self.model_path, subfolder="low_res_scheduler"
            )

            # Create pipeline from components
            self.pipeline = StableDiffusionUpscalePipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                low_res_scheduler=low_res_scheduler,
            )

            # Enable memory optimizations based on environment settings
            logger.info("Enabling memory optimizations...")

            if self.enable_attention_slicing:
                logger.debug("Enabling attention slicing")
                self.pipeline.enable_attention_slicing()

            if self.enable_xformers:
                logger.debug("Enabling xformers memory efficient attention")
                self.pipeline.unet.set_attn_processor(AttnProcessor2_0())
                self.pipeline.enable_xformers_memory_efficient_attention(
                    attention_op=MemoryEfficientAttentionFlashAttentionOp
                )
                self.pipeline.vae.enable_xformers_memory_efficient_attention(
                    attention_op=None
                )

            if self.enable_vae_tiling:
                logger.debug("Enabling VAE tiling")
                self.pipeline.vae.enable_tiling()

            if self.enable_cpu_offload:
                logger.debug("Enabling model CPU offload")
                self.pipeline.enable_model_cpu_offload()

        logger.info("Pipeline loaded successfully with memory optimizations enabled")

    def upscale_image(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str,
        output_path: str,
        down_scale: float = 1.5,
        guidance_scale: float = 9.5,
        num_inference_steps: int = 75,
        skip_existing: bool = True,
    ) -> bool:
        """
        Upscale a single image.

        Args:
            image_path: Path to input image
            prompt: Text prompt for upscaling
            negative_prompt: Negative text prompt
            output_path: Path for output image
            down_scale: Factor to downscale input before upscaling
            guidance_scale: Guidance scale for diffusion
            num_inference_steps: Number of denoising steps
            skip_existing: Skip processing if output already exists

        Returns:
            True if successful, False otherwise
        """
        if skip_existing and os.path.exists(output_path):
            logger.info(f"Output {output_path} already exists, skipping")
            return True

        try:
            logger.info(f"Processing {image_path}")

            # Load and preprocess image
            with open(image_path, "rb") as f:
                content = f.read()

            low_res_img = Image.open(image_path).convert("RGB")
            width, height = low_res_img.size
            low_res_img = low_res_img.resize(
                (int(width / down_scale), int(height / down_scale))
            )

            # Run upscaling
            with torch.amp.autocast("cuda", dtype=self.dtype):
                upscaled_image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=low_res_img,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                ).images[0]

            # Save result
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            upscaled_image.save(output_path)
            logger.info(f"Saved upscaled image to {output_path}")

            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()

            return True

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return False


def load_prompts_from_file(prompt_file: str) -> Tuple[str, str]:
    """
    Load prompts from file.

    Args:
        prompt_file: Path to file containing prompts separated by '--n'

    Returns:
        Tuple of (prompt, negative_prompt)
    """
    try:
        with open(prompt_file, "r") as f:
            content = f.read().strip()
            if "--n" in content:
                prompt, neg_prompt = content.split("--n", 1)
                return prompt.strip(), neg_prompt.strip()
            else:
                return content, ""
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_file}")
        raise click.ClickException(f"Prompt file not found: {prompt_file}")


def get_image_files(
    input_paths: List[str], filter_regex: Optional[str] = None, recursive: bool = True
) -> List[str]:
    """
    Get list of image files from input paths (can be files or directories).

    Args:
        input_paths: List of file or directory paths
        filter_regex: Optional regex pattern to filter filenames
        recursive: Whether to search directories recursively

    Returns:
        List of image file paths
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    image_files = []

    # Compile regex pattern if provided
    regex_pattern = None
    if filter_regex:
        try:
            regex_pattern = re.compile(filter_regex, re.IGNORECASE)
            logger.info(f"Using regex filter: {filter_regex}")
        except re.error as e:
            logger.error(f"Invalid regex pattern '{filter_regex}': {e}")
            raise click.ClickException(f"Invalid regex pattern '{filter_regex}': {e}")

    for path in input_paths:
        path_obj = Path(path)
        if path_obj.is_file():
            if path_obj.suffix.lower() in image_extensions:
                # Apply regex filter if specified
                if regex_pattern is None or regex_pattern.search(path_obj.name):
                    image_files.append(str(path_obj))
        elif path_obj.is_dir():
            logger.info(f"Scanning directory: {path_obj}")
            search_method = path_obj.rglob if recursive else path_obj.glob

            for ext in image_extensions:
                # Search for files with this extension
                pattern = f"*{ext}" if recursive else f"*{ext}"
                found_files = list(search_method(pattern))
                found_files.extend(list(search_method(pattern.upper())))

                # Apply regex filter to found files
                for file_path in found_files:
                    if regex_pattern is None or regex_pattern.search(file_path.name):
                        image_files.append(str(file_path))
        else:
            logger.warning(f"Path does not exist: {path_obj}")

    unique_files = list(
        dict.fromkeys(image_files)
    )  # Remove duplicates while preserving order
    logger.info(f"Found {len(unique_files)} image files")

    return sorted(unique_files)


@click.command()
@click.option(
    "--model-path",
    "-m",
    default=lambda: os.getenv(
        "UPSCALE_MODEL_PATH", "stabilityai/stable-diffusion-x4-upscaler"
    ),
    help="Path to the Stable Diffusion x4 upscaler model",
)
@click.option(
    "--input",
    "-i",
    "input_paths",
    multiple=True,
    required=True,
    help="Input image file(s) or directory. Can be specified multiple times.",
)
@click.option(
    "--output-dir",
    "-o",
    default=lambda: os.getenv("UPSCALE_OUTPUT_DIR", "./upscaled"),
    help="Output directory for upscaled images",
)
@click.option(
    "--prompt",
    "-p",
    default=lambda: os.getenv("UPSCALE_DEFAULT_PROMPT"),
    help="Text prompt for upscaling (overrides prompt file)",
)
@click.option(
    "--negative-prompt",
    "-n",
    default=lambda: os.getenv("UPSCALE_DEFAULT_NEGATIVE_PROMPT", ""),
    help="Negative text prompt",
)
@click.option(
    "--prompt-file",
    default=lambda: os.getenv("UPSCALE_PROMPT_FILE"),
    help="File containing prompts separated by --n",
)
@click.option(
    "--down-scale",
    default=lambda: float(os.getenv("UPSCALE_DOWN_SCALE", "1.5")),
    type=float,
    help="Factor to downscale input before upscaling",
)
@click.option(
    "--guidance-scale",
    default=lambda: float(os.getenv("UPSCALE_GUIDANCE_SCALE", "9.5")),
    type=float,
    help="Guidance scale for diffusion (higher = more prompt adherence)",
)
@click.option(
    "--steps",
    default=lambda: int(os.getenv("UPSCALE_STEPS", "75")),
    type=int,
    help="Number of inference steps",
)
@click.option(
    "--skip-existing/--overwrite",
    default=True,
    help="Skip processing if output file already exists",
)
@click.option(
    "--filter",
    "-f",
    default=lambda: os.getenv("UPSCALE_DEFAULT_FILTER"),
    help="Regex pattern to filter input files (case-insensitive)",
)
@click.option(
    "--recursive/--no-recursive",
    default=lambda: os.getenv("UPSCALE_RECURSIVE", "true").lower() == "true",
    help="Search directories recursively",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(
    model_path: str,
    input_paths: Tuple[str, ...],
    output_dir: str,
    prompt: Optional[str],
    negative_prompt: str,
    prompt_file: Optional[str],
    down_scale: float,
    guidance_scale: float,
    steps: int,
    skip_existing: bool,
    filter: Optional[str],
    recursive: bool,
    verbose: bool,
):
    """
    Upscale images using Stable Diffusion x4 upscaler with memory optimization.

    Examples:
        upscale_4x.py -i image.png -p "high quality, detailed"
        upscale_4x.py -i ./images/ --prompt-file prompts.txt
        upscale_4x.py -i img1.png img2.png -o ./results/
        upscale_4x.py -i ./photos/ -f "portrait.*\.jpg" -p "professional headshot"
        upscale_4x.py -i ./dataset/ -f "^img_\d{4}" --no-recursive
    """
    # Set log level from environment or CLI
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        log_level = os.getenv("UPSCALE_LOG_LEVEL", "INFO").upper()
        numeric_level = getattr(logging, log_level, logging.INFO)
        logging.getLogger().setLevel(numeric_level)

    # Determine prompts
    if prompt is None:
        if prompt_file is None:
            prompt_file = os.getenv("UPSCALE_PROMPT_FILE")
            if prompt_file is None:
                raise click.ClickException(
                    "Either --prompt or --prompt-file must be specified, "
                    "or UPSCALE_PROMPT_FILE environment variable must be set"
                )
        prompt, file_neg_prompt = load_prompts_from_file(prompt_file)
        if not negative_prompt:
            negative_prompt = file_neg_prompt

    # Get image files
    image_files = get_image_files(
        list(input_paths), filter_regex=filter, recursive=recursive
    )
    if not image_files:
        filter_msg = f" matching filter '{filter}'" if filter else ""
        raise click.ClickException(
            f"No image files found{filter_msg} in: {', '.join(input_paths)}"
        )

    logger.info(f"Found {len(image_files)} images to process")
    logger.info(f"Using prompt: {prompt}")
    if negative_prompt:
        logger.info(f"Using negative prompt: {negative_prompt}")

    # Initialize pipeline
    upscaler = UpscalePipeline(model_path)
    upscaler.load_pipeline()

    # Process images
    successful = 0
    failed = 0

    for image_path in image_files:
        # Generate output path
        image_name = Path(image_path).name
        output_path = os.path.join(output_dir, f"upscaled_{image_name}")

        success = upscaler.upscale_image(
            image_path=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_path=output_path,
            down_scale=down_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            skip_existing=skip_existing,
        )

        if success:
            successful += 1
        else:
            failed += 1

    logger.info(f"Processing complete: {successful} successful, {failed} failed")

    if failed > 0:
        raise click.ClickException(f"Failed to process {failed} images")


if __name__ == "__main__":
    main()
