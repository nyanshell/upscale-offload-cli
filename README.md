# Stable Diffusion 4x Upscaler

CLI tool for upscaling images using [Stable Diffusion x4 upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler).

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd upscale-offload
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

## Quick Start

### Basic Usage

```bash
# Upscale a single image
python upscale_4x.py -i image.png -p "high quality, detailed"

# Process a directory with prompt file
python upscale_4x.py -i ./images/ --prompt-file prompts.txt

# Multiple inputs with custom output
python upscale_4x.py -i img1.png img2.png -o ./results/
```

### Advanced Usage

```bash
# Filter JPG portraits in photos directory
./upscale_4x.py -i ./photos/ -f "portrait.*\.jpg" -p "professional headshot"

# Filter files with date pattern (non-recursive)
./upscale_4x.py -i ./dataset/ -f "2025\d{4}_.*\.png" --no-recursive

# Custom parameters
./upscale_4x.py -i ./images/ -p "artistic masterpiece" \
  --guidance-scale 12 --steps 100 --down-scale 2.0
```

## Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`) to configure defaults:

```bash
# Model configuration
UPSCALE_MODEL_PATH=/path/to/stable-diffusion-x4-upscaler
# or use HuggingFace: stabilityai/stable-diffusion-x4-upscaler

# Default prompts
UPSCALE_DEFAULT_PROMPT="high quality, detailed, best quality, ultra high res, 8k"
UPSCALE_DEFAULT_NEGATIVE_PROMPT="blurry, low quality, pixelated, artifacts"

# Processing defaults
UPSCALE_OUTPUT_DIR=./upscaled
UPSCALE_DOWN_SCALE=1.5
UPSCALE_GUIDANCE_SCALE=9.5
UPSCALE_STEPS=75

# Performance settings
UPSCALE_DTYPE=bfloat16
UPSCALE_ENABLE_VAE_TILING=true
UPSCALE_ENABLE_CPU_OFFLOAD=true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model-path` | `-m` | Path to upscaler model | From env or HuggingFace |
| `--input` | `-i` | Input files/directories | Required |
| `--output-dir` | `-o` | Output directory | `./upscaled` |
| `--prompt` | `-p` | Text prompt | From env |
| `--negative-prompt` | `-n` | Negative prompt | From env |
| `--prompt-file` | | Prompt file path | From env |
| `--down-scale` | | Downscale factor | `1.5` |
| `--guidance-scale` | | Guidance scale | `9.5` |
| `--steps` | | Inference steps | `75` |
| `--filter` | `-f` | Regex filter pattern | None |
| `--recursive` | | Search recursively | `true` |
| `--skip-existing` | | Skip existing files | `true` |
| `--verbose` | `-v` | Verbose logging | `false` |

## Examples

### Prompt Files

Create a text file with prompts separated by `--n`:

```
high quality, detailed, masterpiece, 8k resolution
--n
blurry, low quality, artifacts, distorted
```

### Regex Filtering

```bash
# Only process files containing "portrait"
--filter "portrait"

# Files starting with "img_" followed by 4 digits
--filter "^img_\d{4}"

# Multiple patterns
--filter "(test|sample|demo)"

# Specific file extensions
--filter ".*\.(jpg|png)$"
```

## License

MIT License
