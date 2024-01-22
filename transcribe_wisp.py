# pip3 install transformers
# pip3 install llama-cpp-python
# pip3 install huggingface-hub sentence-transformers langchain
# pip3 install ffmpeg

# Downalod Models
# huggingface-cli download TheBloke/Llama-2-7b-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False


from transformers import pipeline
import torch
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(device)
transcriber = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base.en", device=device
)

import sys


def transcribe(chunk_length_s=15.0, stream_chunk_s=1.0):
    sampling_rate = transcriber.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        
        stream_chunk_s=stream_chunk_s,
    )

    print("Start speaking...")
    for item in transcriber(mic, generate_kwargs={"max_new_tokens": 128}):
        sys.stdout.write("\033[K")
        print(item["text"], end="\r")
        if not item["partial"][0]:
            break


transcribe()
