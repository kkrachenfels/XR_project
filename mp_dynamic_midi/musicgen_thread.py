from transformers import MusicgenForConditionalGeneration, MusicgenProcessor
from musicgen_utils import MusicgenStreamer
import torch
from threading import Thread
from queue import Queue
import numpy as np
import sounddevice as sd


# def build_prompt_from_command(command):
#     """
#     Constructs a basic prompt string for MusicGen based on gesture-based command parameters.
#     """
#     key = command.get('key', 'C')
#     progression = command.get('progression', 'major')
#     tempo_factor = command.get('tempo', 1.0)

#     # Convert factor into approximate BPM
#     base_bpm = 60
#     bpm = int(base_bpm / tempo_factor) if tempo_factor else base_bpm

#     return f"A {progression} piece in {key} major, approximately {bpm} BPM with expressive instrumentals."
def build_prompt_from_command(command):
    """
    Builds an LMA-aligned MusicGen prompt using:
    - key (Shape/Emotion),
    - progression (Shape/Relationship),
    - tempo (Effort/Dynamics: Time & Weight).
    """

    key = command.get('key', 'C')
    progression = command.get('progression', 'major').lower()
    tempo = command.get('tempo', 1.0)

    # === Effort/Dynamics via Tempo ===
    if tempo < 0.5:
        effort_time = "very quick and intense"
        weight = "strong and urgent"
        energy = "explosive energy and momentum"
    elif tempo < 0.85:
        effort_time = "quick and animated"
        weight = "moderately strong"
        energy = "lively and engaging motion"
    elif tempo < 1.15:
        effort_time = "steady and natural"
        weight = "balanced and fluid"
        energy = "flowing and expressive energy"
    elif tempo < 1.5:
        effort_time = "slow and intentional"
        weight = "light and gentle"
        energy = "delicate, sustained movement"
    else:
        effort_time = "very slow and suspended"
        weight = "weightless and airy"
        energy = "floating, meditative quality"

    # === Shape/Emotion via Progression ===
    if progression == "major":
        mood = "bright, open, and uplifting"
        shape_qual = "expanding and confident"
    elif progression == "minor":
        mood = "emotional, introspective, and tender"
        shape_qual = "contracting and inward-focused"
    else:
        mood = "ambiguous, atmospheric, and evolving"
        shape_qual = "shifting or fluid"

    # === Final Prompt ===
    return (
        f"A {mood} composition in {key} {progression}. "
        f"The tempo evokes {effort_time} movement, with {weight} phrasing and a sense of {energy}. "
        f"The musical contour suggests a {shape_qual} gesture."
    )




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

        max_new_tokens = int(frame_rate * 5)  # 10 seconds audio
        play_steps = int(frame_rate * 2.5)     # stream every 1 seconds

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
                    'sampling_rate': sampling_rate,
                    'tempo': command.get('tempo', 1.0),
                    'progression': command.get('progression', 'major'),
                    'key': command.get('key', 'C')
                })

        except Exception as e:
            print(f"[MusicGen] Error while streaming: {e}")
            continue

        thread.join()

    print("[MusicGen] Music thread closed.")
