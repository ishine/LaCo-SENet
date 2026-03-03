import random
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np

def random_sample(noisy, clean, segment):
    length = noisy.shape[-1]
    if length >= segment:
        max_audio_start = length - segment
        rand_num = random.random()
        if rand_num < 0.01:
            audio_start = 0
        elif rand_num < 0.02:
            audio_start = max_audio_start
        else:
            audio_start = random.randint(0, max_audio_start)
        noisy = noisy[audio_start: audio_start + segment]
        clean = clean[audio_start: audio_start + segment]
    else:
        noisy = F.pad(noisy, (0, segment - length), mode='constant')
        clean = F.pad(clean, (0, segment - length), mode='constant')
    return noisy, clean


def segment_sample(noisy, clean, segment):
    """Split 1D audio into [N, segment] batched segments for validation.

    Returns 2D tensors [num_segments, segment] with the last segment
    zero-padded to full segment length if needed.
    """
    length = noisy.shape[-1]
    if length >= segment:
        num_segments = length // segment
        last_segment_size = length % segment
        noisy_segments = [noisy[i*segment: (i+1)*segment] for i in range(num_segments)]
        clean_segments = [clean[i*segment: (i+1)*segment] for i in range(num_segments)]
        if last_segment_size > 0:
            noisy_segments.append(F.pad(noisy[num_segments*segment:], (0, segment - last_segment_size)))
            clean_segments.append(F.pad(clean[num_segments*segment:], (0, segment - last_segment_size)))
        noisy = torch.stack(noisy_segments, dim=0)
        clean = torch.stack(clean_segments, dim=0)
    else:
        noisy = F.pad(noisy, (0, segment - length), mode='constant').unsqueeze(0)
        clean = F.pad(clean, (0, segment - length), mode='constant').unsqueeze(0)
    return noisy, clean


class VoiceBankDataset(torch.utils.data.Dataset):
    def __init__(self, datapair_list, segment=None, with_id=False, with_text=False):
        self.segment = segment
        self.with_id = with_id
        self.with_text = with_text
        self.sampling_rate = 16000
        self.audio_pairs = []

        for item in datapair_list:
            id = item["id"]
            noisy = item["noisy"]["array"].astype("float32")
            clean = item["clean"]["array"].astype("float32")
            # Power normalization
            norm_factor = np.sqrt(noisy.shape[-1] / np.sum(noisy ** 2.0))
            noisy = noisy * norm_factor
            clean = clean * norm_factor
            self.audio_pairs.append((noisy, clean, id))

    def __len__(self):
        return len(self.audio_pairs)

    def __getitem__(self, index):
        noisy, clean, id = self.audio_pairs[index]

        noisy = torch.FloatTensor(noisy)
        clean = torch.FloatTensor(clean)

        if self.segment is not None:
            noisy, clean = random_sample(noisy, clean, self.segment)

        if self.with_text:
            return noisy, clean, id, ""
        elif self.with_id:
            return noisy, clean, id
        else:
            return noisy, clean


class StepSampler(torch.utils.data.Sampler):
    def __init__(self, length, step):
        # Save the total length and sampling step
        self.step = step
        self.length = length

    def __iter__(self):
        # Return indices at intervals of step
        return iter(range(0, self.length, self.step))

    def __len__(self):
        # Length is how many indices we can produce based on the step
        return self.length // self.step