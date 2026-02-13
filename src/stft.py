import torch

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
    wav = torch.istft(com, n_fft=n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

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