import torch

# Naming convention: `com` = compressed complex spectrogram (mag·cos, mag·sin)
# stacked along stack_dim.  Used throughout the codebase as the primary
# representation passed into / returned from the Backbone model.


def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True, stack_dim=-1):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9))
    # Add epsilon to prevent gradient explosion in atan2 backward
    # When magnitude is very small (near silence), atan2 gradient can explode
    pha = torch.atan2(stft_spec[:, :, :, 1] + 1e-8, stft_spec[:, :, :, 0] + 1e-8)
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=stack_dim)

    return mag, pha, com

def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)

    if center:
        wav = torch.istft(com, n_fft=n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=True)
    else:
        # torch.istft(center=False) fails COLA check with Hann window.
        # Use manual OLA reconstruction instead.
        frames = torch.fft.irfft(com.transpose(-2, -1), n=n_fft)  # [B, T, n_fft]
        frames = frames[..., :win_size]  # [B, T, win_size]
        frames = frames * hann_window
        window_sq = hann_window * hann_window

        B, T = frames.shape[:2]
        output_len = (T - 1) * hop_size + win_size

        buf = torch.zeros(B, output_len, device=com.device, dtype=frames.dtype)
        norm = torch.zeros(output_len, device=com.device, dtype=frames.dtype)

        for t in range(T):
            start = t * hop_size
            buf[:, start:start + win_size] += frames[:, t]
            norm[start:start + win_size] += window_sq

        safe_mask = norm > 1e-8
        buf[:, safe_mask] = buf[:, safe_mask] / norm[safe_mask]
        wav = buf

    return wav

def complex_to_mag_pha(com, stack_dim=-1):
    real, imag = com.chunk(2, dim=stack_dim)
    mag = torch.sqrt(real**2 + imag**2 + 1e-8).squeeze(stack_dim)
    # Add epsilon to prevent gradient explosion in atan2 backward
    pha = torch.atan2(imag + 1e-8, real + 1e-8).squeeze(stack_dim)
    return mag, pha

def mag_pha_to_complex(mag, pha, stack_dim=-1):
    real = mag * torch.cos(pha)
    imag = mag * torch.sin(pha)
    com = torch.stack((real, imag), dim=stack_dim)
    return com


def manual_istft_ola(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0,
                     ola_buffer=None, ola_norm=None):
    """
    Manual iSTFT with cross-chunk OLA buffer accumulation.

    Performs frame-by-frame iFFT with Hann synthesis window and accumulates
    into an OLA buffer. Supports carry-over from previous chunks via
    ola_buffer/ola_norm arguments.

    Args:
        mag: [B, F, T] compressed magnitude
        pha: [B, F, T] phase
        n_fft: FFT size
        hop_size: STFT hop size in samples
        win_size: Window size
        compress_factor: magnitude decompression exponent
        ola_buffer: [ola_size] carry-over OLA buffer from previous chunk (None = zeros)
        ola_norm: [ola_size] carry-over normalization buffer from previous chunk (None = zeros)

    Returns:
        output: [T * hop_size] mature samples (fully normalized)
        new_ola_buffer: [win_size - hop_size] tail carry-over for next chunk
        new_ola_norm: [win_size - hop_size] norm tail carry-over for next chunk
    """
    # Magnitude decompression
    mag = torch.pow(mag, (1.0 / compress_factor))

    # Complex spectrum reconstruction → iFFT
    spec = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))  # [B, F, T]
    # irfft expects [..., F] → transpose to [B, T, F]
    frames = torch.fft.irfft(spec.transpose(1, 2), n=n_fft)  # [B, T, n_fft]

    # Take first win_size samples (when n_fft == win_size this is identity)
    frames = frames[:, :, :win_size]  # [B, T, win_size]

    # Synthesis window
    window = torch.hann_window(win_size, device=mag.device, dtype=mag.dtype)
    frames = frames * window  # [B, T, win_size]
    window_sq = window * window

    B, T, _ = frames.shape
    output_samples = T * hop_size
    tail_size = win_size - hop_size
    total_size = output_samples + tail_size

    # Initialize OLA accumulation buffer
    buf = torch.zeros(total_size, device=mag.device, dtype=mag.dtype)
    norm = torch.zeros(total_size, device=mag.device, dtype=mag.dtype)

    # Prepend carry-over from previous chunk
    if ola_buffer is not None:
        carry_len = min(len(ola_buffer), tail_size)
        buf[:carry_len] = ola_buffer[:carry_len].to(buf.device)
    if ola_norm is not None:
        carry_len = min(len(ola_norm), tail_size)
        norm[:carry_len] = ola_norm[:carry_len].to(norm.device)

    # OLA accumulation (batch dim squeezed — streaming is B=1)
    frames_0 = frames[0]  # [T, win_size]
    for t in range(T):
        start = t * hop_size
        buf[start:start + win_size] += frames_0[t]
        norm[start:start + win_size] += window_sq

    # Extract mature samples (fully overlapped region)
    output = buf[:output_samples]
    output_norm = norm[:output_samples]
    # Normalize where norm > 0 to avoid division by zero (e.g., first chunk edges)
    safe_mask = output_norm > 1e-8
    output[safe_mask] = output[safe_mask] / output_norm[safe_mask]

    # Carry-over tail for next chunk
    new_ola_buffer = buf[output_samples:].clone()
    new_ola_norm = norm[output_samples:].clone()

    return output, new_ola_buffer, new_ola_norm