import pyaudio
import requests

url = "http://localhost:8000/tts"
url1 = "http://localhost:8000/tts_pages/1"

# Parameters
body = {
    "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism.",
}
params = {"speed": 1.5}
params_pages = {"pages": [2, 3, 4, 5], "speed": 1.5}


def audio():
    # Set parameters for audio playback
    sample_rate = 22050
    channels = 1  # Mono
    format = pyaudio.paInt16
    # frames_per_buffer = 1024

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open audio stream for playback
    stream = p.open(format=format, channels=channels, rate=sample_rate, output=True)

    try:
        # Write raw audio to stdout as it's produced
        for audio_bytes in requests.post(
            url, json=body, params=params, stream=True
        ).iter_content(chunk_size=32):
            stream.write(audio_bytes)
    except Exception as e:
        print("Error during audio playback:", e)
    finally:
        # Clean up
        stream.stop_stream()
        stream.close()
        p.terminate()


# Example usage:
audio()
