# from queue import Queue
# from transformers import MusicgenForConditionalGeneration
# from transformers.generation.streamers import BaseStreamer
# import numpy as np
# import torch
# import sounddevice as sd

# class MusicgenStreamer(BaseStreamer):
#     def __init__(
#         self,
#         model: MusicgenForConditionalGeneration,
#         device: str = None,
#         play_steps: int = 10,
#         stride: int = None,
#         timeout: float = None,
#     ):
#         self.decoder = model.decoder
#         self.audio_encoder = model.audio_encoder
#         self.generation_config = model.generation_config
#         self.device = device if device is not None else model.device

#         self.play_steps = play_steps
#         if stride is not None:
#             self.stride = stride
#         else:
#             hop_length = np.prod(self.audio_encoder.config.upsampling_ratios)
#             self.stride = hop_length * (play_steps - self.decoder.num_codebooks) // 6
#         self.token_cache = None
#         self.to_yield = 0

#         self.audio_queue = Queue()
#         self.stop_signal = None
#         self.timeout = timeout

#     def apply_delay_pattern_mask(self, input_ids):
#         _, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
#             input_ids[:, :1],
#             pad_token_id=self.generation_config.decoder_start_token_id,
#             max_length=input_ids.shape[-1],
#         )
#         input_ids = self.decoder.apply_delay_pattern_mask(input_ids, decoder_delay_pattern_mask)
#         input_ids = input_ids[input_ids != self.generation_config.pad_token_id].reshape(
#             1, self.decoder.num_codebooks, -1
#         )
#         input_ids = input_ids[None, ...]
#         input_ids = input_ids.to(self.audio_encoder.device)
#         output_values = self.audio_encoder.decode(input_ids, audio_scales=[None])
#         audio_values = output_values.audio_values[0, 0]
#         return audio_values.cpu().float().numpy()

#     def put(self, value):
#         batch_size = value.shape[0] // self.decoder.num_codebooks
#         if batch_size > 1:
#             raise ValueError("MusicgenStreamer only supports batch size 1")
#         if self.token_cache is None:
#             self.token_cache = value
#         else:
#             self.token_cache = torch.concatenate([self.token_cache, value[:, None]], dim=-1)

#         if self.token_cache.shape[-1] % self.play_steps == 0:
#             audio_values = self.apply_delay_pattern_mask(self.token_cache)
#             self.on_finalized_audio(audio_values[self.to_yield : -self.stride])
#             self.to_yield += len(audio_values) - self.to_yield - self.stride

#     def end(self):
#         if self.token_cache is not None:
#             audio_values = self.apply_delay_pattern_mask(self.token_cache)
#         else:
#             audio_values = np.zeros(self.to_yield)
#         self.on_finalized_audio(audio_values[self.to_yield :], stream_end=True)

#     def on_finalized_audio(self, audio: np.ndarray, stream_end: bool = False):
#         self.audio_queue.put(audio, timeout=self.timeout)
#         if stream_end:
#             self.audio_queue.put(self.stop_signal, timeout=self.timeout)

#     def __iter__(self):
#         return self

#     def __next__(self):
#         value = self.audio_queue.get(timeout=self.timeout)
#         if not isinstance(value, np.ndarray) and value == self.stop_signal:
#             raise StopIteration()
#         else:
#             return value
from queue import Queue
from transformers import MusicgenForConditionalGeneration
from transformers.generation.streamers import BaseStreamer
import numpy as np
import torch

class MusicgenStreamer(BaseStreamer):
    def __init__(
        self,
        model: MusicgenForConditionalGeneration,
        device: str = None,
        play_steps: int = 10,
        stride: int = None,
        timeout: float = None,
    ):
        self.decoder = model.decoder
        self.audio_encoder = model.audio_encoder
        self.generation_config = model.generation_config
        self.device = device if device is not None else model.device

        self.play_steps = play_steps
        if stride is not None:
            self.stride = stride
        else:
            hop_length = np.prod(self.audio_encoder.config.upsampling_ratios)
            self.stride = hop_length * (play_steps - self.decoder.num_codebooks) // 6
        self.token_cache = None
        self.to_yield = 0

        self.audio_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def apply_delay_pattern_mask(self, input_ids):
        _, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids[:, :1],
            pad_token_id=self.generation_config.decoder_start_token_id,
            max_length=input_ids.shape[-1],
        )
        input_ids = self.decoder.apply_delay_pattern_mask(input_ids, decoder_delay_pattern_mask)
        input_ids = input_ids[input_ids != self.generation_config.pad_token_id].reshape(
            1, self.decoder.num_codebooks, -1
        )
        input_ids = input_ids[None, ...]
        input_ids = input_ids.to(self.audio_encoder.device)
        output_values = self.audio_encoder.decode(input_ids, audio_scales=[None])
        audio_values = output_values.audio_values[0, 0]
        
        # Robust conversion to numpy array
        try:
            if hasattr(audio_values, 'cpu'):
                # It's a PyTorch tensor
                return audio_values.cpu().float().numpy()
            elif isinstance(audio_values, torch.Tensor):
                # It's a tensor but maybe already on CPU
                return audio_values.float().numpy()
            else:
                # It's already a numpy array
                return np.array(audio_values, dtype=np.float32)
        except Exception as e:
            print(f"[MusicgenStreamer] Error converting audio_values: {e}")
            print(f"[MusicgenStreamer] audio_values type: {type(audio_values)}")
            # Fallback: try direct conversion
            return np.array(audio_values).astype(np.float32)

    def put(self, value):
        batch_size = value.shape[0] // self.decoder.num_codebooks
        if batch_size > 1:
            raise ValueError("MusicgenStreamer only supports batch size 1")
        if self.token_cache is None:
            self.token_cache = value
        else:
            self.token_cache = torch.concatenate([self.token_cache, value[:, None]], dim=-1)

        if self.token_cache.shape[-1] % self.play_steps == 0:
            try:
                audio_values = self.apply_delay_pattern_mask(self.token_cache)
                self.on_finalized_audio(audio_values[self.to_yield : -self.stride])
                self.to_yield += len(audio_values) - self.to_yield - self.stride
            except Exception as e:
                print(f"[MusicgenStreamer] Error in put(): {e}")
                raise

    def end(self):
        try:
            if self.token_cache is not None:
                audio_values = self.apply_delay_pattern_mask(self.token_cache)
            else:
                audio_values = np.zeros(self.to_yield)
            self.on_finalized_audio(audio_values[self.to_yield :], stream_end=True)
        except Exception as e:
            print(f"[MusicgenStreamer] Error in end(): {e}")
            # Send empty audio to end the stream
            self.on_finalized_audio(np.zeros(1024), stream_end=True)

    def on_finalized_audio(self, audio: np.ndarray, stream_end: bool = False):
        self.audio_queue.put(audio, timeout=self.timeout)
        if stream_end:
            self.audio_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.audio_queue.get(timeout=self.timeout)
        if not isinstance(value, np.ndarray) and value == self.stop_signal:
            raise StopIteration()
        else:
            return value