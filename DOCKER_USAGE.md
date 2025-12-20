# WhisperX Docker Usage Guide

This project is set up to run WhisperX with Speaker Diarization inside a Docker container, fully accelerated by NVIDIA GPUs.

## Prerequisites

1.  **NVIDIA Drivers**: Ensure you have the latest drivers installed for your GPU.
2.  **Docker Desktop (Windows)** or **Docker Engine (Linux)**.
3.  **NVIDIA Container Toolkit**:
    *   **Windows**: Included in Docker Desktop (ensure WSL2 backend is selected).
    *   **Linux**: Must be installed separately to allow Docker to access the GPU.

## Setup

1.  **Hugging Face Token**:
    *   You need a Hugging Face token with access to `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.
    *   Accept the user agreements on those model pages on Hugging Face.
    *   Create a `.env` file in this directory (copy from example if available, or just create it):
        ```env
        HF_TOKEN=hf_your_token_here
        ```

2.  **Build the Container**:
    ```bash
    docker compose build
    ```

## Usage

### 1. Place Input Files
Put your audio/video files (mp3, wav, mp4, etc.) in the `input/` directory. This directory is mounted to `/data` inside the container.

### 2. Run Transcription
Run the container on a specific file. The path inside the container is `/data/filename.ext`.

```bash
docker compose run --rm whisperx /data/your_audio_file.mp3
```

### 3. Output
The transcription results (transcript, segments, etc.) will be saved in the `input/` directory alongside your original file.

### 4. Batch Transcription (Recursive)
To transcribe an entire directory structure recursively:
1.  Ensure your `batch_transcribe.py` is in the project root.
2.  Run the script, pointing to your source directory:
    ```bash
    python batch_transcribe.py "C:\Path\To\My\Audio\Collection"
    ```
    *   It will find all audio files.
    *   Copy them one-by-one to the container input.
    *   Transcribe them.
    *   Move the transcript back to the original folder.
    *   Cleanup temporary files.

## Troubleshooting

*   **GPU Not Found**: Run `docker compose run --rm whisperx nvidia-smi` to verify the container can see your GPU.
*   **Permissions**: Ensure your `.env` file is readable.
*   **Rebuild**: If you change `Dockerfile` or `pyproject.toml`, always run `docker compose build` again.
