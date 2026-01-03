# Models Directory

This directory stores the local AI models used by Bookipedia-AI, specifically the Piper TTS model.

## Piper TTS Model

The project is configured to use the `en_US-amy-medium` voice model. You need to download the `.onnx` model file and its corresponding `.json` configuration file.

### Download Instructions

1.  **Download the ONNX model:**
    [en_US-amy-medium.onnx](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx)

2.  **Download the JSON config:**
    [en_US-amy-medium.onnx.json](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json)

3.  **Place files:**
    Move both downloaded files into this `models/` directory.

    Your directory structure should look like this:

    ```
    bookipedia/
    └── models/
        ├── en_US-amy-medium.onnx
        ├── en_US-amy-medium.onnx.json
        └── README.md
    ```

### More Voices

You can find more voices and languages in the [Piper Voices Documentation](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md).
If you choose a different voice, make sure to update the `PIPER_MODEL_PATH` in `src/bookipedia_ai/utils/config.py`.

## ONNX Runtime Setup

The Piper TTS engine depends on `onnxruntime`.

*   **CPU:** The standard `piper-tts` installation includes `onnxruntime` for CPU execution.
*   **GPU (CUDA):** To enable GPU acceleration, you must uninstall `onnxruntime` and install `onnxruntime-gpu`.

    ```bash
    pip uninstall onnxruntime
    pip install onnxruntime-gpu
    ```

    *Note: Ensure your CUDA and cuDNN versions are compatible with the installed `onnxruntime-gpu` version.*

## Other Models

Other models (like OCR and Embedding models) are typically downloaded automatically by their respective libraries (docTR, Hugging Face Transformers) upon first use and cached in the standard system cache directories (e.g., `~/.cache/`), so they do not need to be manually placed here.
