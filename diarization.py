from itertools import islice

import torch
from deepspeech_training.util.audio import read_frames_from_file, pcm_to_np


class Diarization:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.diarization_pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')
        self.frame_duration_ms = 30
        self.speakers = []

    def generate_values(self):
        diarization = self.diarization_pipeline({'audio': self.audio_path})
        iter_frames = read_frames_from_file(self.audio_path)

        frame_offset = 0
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            # Convert to ms
            start_ms = int(segment.start * 1000)
            # Round down start to next 30ms
            start = start_ms - (start_ms % self.frame_duration_ms)

            # Convert to ms
            end_ms = int(segment.end * 1000)
            # Round up end to next 30ms
            end = end_ms - (end_ms % -self.frame_duration_ms)

            samples = b"".join(
                islice(
                    iter_frames,
                    start // self.frame_duration_ms - frame_offset,
                    end // self.frame_duration_ms - frame_offset)
            )
            frame_offset = end_ms // self.frame_duration_ms

            samples = pcm_to_np(samples)
            self.speakers.append(speaker)
            yield start, end, samples


if __name__ == "__main__":
    d = Diarization("/container/ma-dueggi/ZzPwRXytr7U_Jesus Calms The Storm.wav")
    print(list(d.generate_values()))
