from transformers import MusicgenForConditionalGeneration, MusicgenProcessor
from musicgen_utils import MusicgenStreamer
import torch
from threading import Thread
from queue import Queue
import numpy as np
import sounddevice as sd


def build_prompt_from_command(command):
    """
    Constructs a basic prompt string for MusicGen based on gesture-based command parameters.
    """
    key = command.get('key', 'C')
    progression = command.get('progression', 'major')
    tempo_factor = command.get('tempo', 1.0)

    # Convert factor into approximate BPM
    base_bpm = 60
    bpm = int(base_bpm / tempo_factor) if tempo_factor else base_bpm

    return f"A {progression} piece in {key} major, approximately {bpm} BPM with expressive instrumentals."


def music_thread_v2(command_queue: Queue, return_queue: Queue):
    print("[MusicGen] Loading model...")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    processor = MusicgenProcessor.from_pretrained("facebook/musicgen-small")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if device == "cuda":
        model.half()

    sampling_rate = model.audio_encoder.config.sampling_rate
    frame_rate = model.audio_encoder.config.frame_rate

    print("[MusicGen] Model initialized and waiting for prompts...")

    while True:
        command = command_queue.get()
        if command.get('type') == 'stop':
            print("[MusicGen] Received stop command. Shutting down...")
            break

        prompt = build_prompt_from_command(command)
        print(f"[MusicGen] Prompt: {prompt}")

        inputs = processor(text=prompt, return_tensors="pt").to(device)

        max_new_tokens = int(frame_rate * 10)  # 10 seconds audio
        play_steps = int(frame_rate * 2.5)     # stream every ~2.5 seconds

        streamer = MusicgenStreamer(model, device=device, play_steps=play_steps)

        thread = Thread(target=model.generate, kwargs={
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
        })
        thread.start()

        try:
            for chunk in streamer:
                #chunk = chunk.cpu().numpy()  # Convert tensor to numpy array for playback
                chunk = chunk.astype(np.float32)
                duration = round(chunk.shape[0] / sampling_rate, 2)
                print(f"[MusicGen] → Streaming chunk ({duration} sec)")

                sd.play(chunk, samplerate=sampling_rate)
                sd.wait()  # Wait for chunk to finish before playing next

                # Optional: still send audio to return_queue if needed elsewhere
                return_queue.put({
                    'audio_chunk': chunk,
                    'sampling_rate': sampling_rate
                })
        except Exception as e:
            print(f"[MusicGen] Error while streaming: {e}")
            continue

        thread.join()

    print("[MusicGen] Music thread closed.")
