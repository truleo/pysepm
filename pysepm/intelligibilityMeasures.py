import numpy as np
import torch
from pystoi import stoi as pystoi  # https://github.com/mpariente/pystoi
from scipy.interpolate import interp1d
from scipy.signal import butter, hilbert, lfilter

from .util import extract_overlapped_windows, resample_matlab_like

stoi = pystoi


def do_stft(input_signal, n_fft, skiprate, window, winlength):
    return torch.stft(
        input_signal,
        n_fft=n_fft,
        hop_length=skiprate,
        window=torch.tensor(window).to(input_signal.device),
        normalized=False,
        center=False,
        onesided=False,
        win_length=winlength,
        return_complex=True,
    )


def fwseg_noise(
    clean, noisy, fs=16000, frameLen=0.03, overlap=0.75, device=torch.device("cuda")
):
    clean_length = len(clean)
    noisy_length = len(noisy)
    rms_all = np.linalg.norm(clean) / np.sqrt(noisy_length)

    winlength = round(frameLen * fs)  # window length in samples
    skiprate = int(np.floor((1 - overlap) * frameLen * fs))  # window skip in samples
    num_crit = 16  # number of critical bands
    n_fft = int(2 ** np.ceil(np.log2(2 * winlength)))
    n_fftby2 = int(n_fft / 2)

    cent_freq = np.zeros((num_crit,))
    bandwidth = np.zeros((num_crit,))

    # ----------------------------------------------------------------------
    # Critical Band Filter Definitions (Center Frequency and Bandwidths in Hz)
    # ----------------------------------------------------------------------
    cent_freq[0] = 150.0000
    bandwidth[0] = 100.0000
    cent_freq[1] = 250.000
    bandwidth[1] = 100.0000
    cent_freq[2] = 350.000
    bandwidth[2] = 100.0000
    cent_freq[3] = 450.000
    bandwidth[3] = 110.0000
    cent_freq[4] = 570.000
    bandwidth[4] = 120.0000
    cent_freq[5] = 700.000
    bandwidth[5] = 140.0000
    cent_freq[6] = 840.000
    bandwidth[6] = 150.0000
    cent_freq[7] = 1000.000
    bandwidth[7] = 160.000
    cent_freq[8] = 1170.000
    bandwidth[8] = 190.000
    cent_freq[9] = 1370.000
    bandwidth[9] = 210.000
    cent_freq[10] = 1600.000
    bandwidth[10] = 240.000
    cent_freq[11] = 1850.000
    bandwidth[11] = 280.000
    cent_freq[12] = 2150.000
    bandwidth[12] = 320.000
    cent_freq[13] = 2500.000
    bandwidth[13] = 380.000
    cent_freq[14] = 2900.000
    bandwidth[14] = 450.000
    cent_freq[15] = 3400.000
    bandwidth[15] = 550.000

    Weight = np.array(
        [
            0.0192,
            0.0312,
            0.0926,
            0.1031,
            0.0735,
            0.0611,
            0.0495,
            0.044,
            0.044,
            0.049,
            0.0486,
            0.0493,
            0.049,
            0.0547,
            0.0555,
            0.0493,
        ]
    )

    # ----------------------------------------------------------------------
    # Set up the critical band filters.  Note here that Gaussianly shaped
    # filters are used.  Also, the sum of the filter weights are equivalent
    # for each critical band filter.  Filter less than -30 dB and set to
    # zero.
    # ----------------------------------------------------------------------
    crit_filter = np.zeros((num_crit, int(n_fftby2)))
    g = np.zeros((num_crit, n_fftby2))

    b = bandwidth
    q = cent_freq / 1000
    p = 4 * 1000 * q / b
    # Eq. (7)

    # 15.625=4000/256
    j = np.arange(0, n_fftby2)

    for i in range(num_crit):
        g[i, :] = np.abs(1 - j * (fs / n_fft) / (q[i] * 1000))
        # Eq. (9)
        crit_filter[i, :] = (1 + p[i] * g[i, :]) * np.exp(-p[i] * g[i, :])
        #  Eq. (8)

    # number of frames
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    hannWin = 0.5 * (
        1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1))
    )

    clean_spec = do_stft(
        torch.tensor(
            clean[0 : int(num_frames) * skiprate + int(winlength - skiprate)],
            device=device,
        ),
        n_fft,
        skiprate,
        hannWin,
        winlength,
    )
    noisy_spec = do_stft(
        torch.tensor(
            noisy[0 : int(num_frames) * skiprate + int(winlength - skiprate)],
            device=device,
        ),
        n_fft,
        skiprate,
        hannWin,
        winlength,
    )

    clean_frames = extract_overlapped_windows(
        clean[0 : int(num_frames) * skiprate + int(winlength - skiprate)],
        winlength,
        winlength - skiprate,
        None,
    )
    # drop frames where we get near the boundaries
    clean_frames = clean_frames[: min(clean_frames.shape[0], clean_spec.shape[-1]), :]
    rms_seg = np.linalg.norm(clean_frames, axis=-1) / np.sqrt(winlength)
    rms_db = 20 * np.log10(rms_seg / rms_all)
    # --------------------------------------------------------------
    # cal r2_high,r2_middle,r2_low
    highInd = np.where(rms_db >= 0)
    highInd = highInd[0]
    middleInd = np.where((rms_db >= -10) & (rms_db < 0))
    middleInd = middleInd[0]
    lowInd = np.where(rms_db < -10)
    lowInd = lowInd[0]

    num_high = torch.sum(
        clean_spec[0:n_fftby2, highInd] * noisy_spec[0:n_fftby2, highInd].conj(),
        axis=-1,
    )
    denx_high = clean_spec[0:n_fftby2, highInd].abs().pow(2).sum(axis=-1)
    deny_high = noisy_spec[0:n_fftby2, highInd].abs().pow(2).sum(axis=-1)

    num_middle = torch.sum(
        clean_spec[0:n_fftby2, middleInd] * noisy_spec[0:n_fftby2, middleInd].conj(),
        axis=-1,
    )
    denx_middle = clean_spec[0:n_fftby2, middleInd].abs().pow(2).sum(axis=-1)
    deny_middle = noisy_spec[0:n_fftby2, middleInd].abs().pow(2).sum(axis=-1)

    num_low = torch.sum(
        clean_spec[0:n_fftby2, lowInd] * noisy_spec[0:n_fftby2, lowInd].conj(), axis=-1
    )
    denx_low = clean_spec[0:n_fftby2, lowInd].abs().pow(2).sum(axis=-1)
    deny_low = noisy_spec[0:n_fftby2, lowInd].abs().pow(2).sum(axis=-1)

    num2_high = num_high.abs().pow(2)
    r2_high = num2_high / (denx_high * deny_high)

    num2_middle = num_middle.abs().pow(2)
    r2_middle = num2_middle / (denx_middle * deny_middle)

    num2_low = num_low.abs().pow(2)
    r2_low = num2_low / (denx_low * deny_low)
    # --------------------------------------------------------------
    # cal distortion frame by frame

    clean_spec = clean_spec.abs().pow(2)
    noisy_spec = noisy_spec.abs().pow(2)

    W_freq = Weight
    crit_filter = torch.tensor(crit_filter, device=clean_spec.device)
    noisy_energy = torch.matmul(
        crit_filter, (noisy_spec[0:n_fftby2, highInd].T * r2_high).T
    )
    de_noisy_energy = torch.matmul(
        crit_filter, (noisy_spec[0:n_fftby2, highInd].T * (1 - r2_high)).T
    )
    SDR = noisy_energy / de_noisy_energy
    # Eq 13 in Kates (2005)
    SDRlog = 10 * SDR.log10()
    SDRlog_lim = SDRlog
    SDRlog_lim[SDRlog_lim < -15] = -15
    SDRlog_lim[SDRlog_lim > 15] = 15  # limit between [-15, 15]
    Tjm = (SDRlog_lim + 15) / 30
    distortionh = W_freq.dot(Tjm.detach().cpu().numpy()) / np.sum(W_freq, axis=0)
    distortionh[distortionh < 0] = 0

    noisy_energy = torch.matmul(
        crit_filter, (noisy_spec[0:n_fftby2, middleInd].T * r2_middle).T
    )
    de_noisy_energy = torch.matmul(
        crit_filter, (noisy_spec[0:n_fftby2, middleInd].T * (1 - r2_middle)).T
    )
    SDR = noisy_energy / de_noisy_energy
    # Eq 13 in Kates (2005)
    SDRlog = 10 * SDR.log10()
    SDRlog_lim = SDRlog
    SDRlog_lim[SDRlog_lim < -15] = -15
    SDRlog_lim[SDRlog_lim > 15] = 15  # limit between [-15, 15]
    Tjm = (SDRlog_lim + 15) / 30
    distortionm = W_freq.dot(Tjm.detach().cpu().numpy()) / np.sum(W_freq, axis=0)
    distortionm[distortionm < 0] = 0

    noisy_energy = torch.matmul(
        crit_filter, (noisy_spec[0:n_fftby2, lowInd].T * r2_low).T
    )
    de_noisy_energy = torch.matmul(
        crit_filter, (noisy_spec[0:n_fftby2, lowInd].T * (1 - r2_low)).T
    )
    SDR = noisy_energy / de_noisy_energy
    # Eq 13 in Kates (2005)
    SDRlog = 10 * SDR.log10()
    SDRlog_lim = SDRlog
    SDRlog_lim[SDRlog_lim < -15] = -15
    SDRlog_lim[SDRlog_lim > 15] = 15  # limit between [-15, 15]
    Tjm = (SDRlog_lim + 15) / 30
    distortionl = W_freq.dot(Tjm.detach().cpu().numpy()) / np.sum(W_freq, axis=0)
    distortionl[distortionl < 0] = 0

    return distortionh, distortionm, distortionl


def csii(clean, noisy, sample_rate):
    sampleLen = min(len(clean), len(noisy))
    clean = clean[0:sampleLen]
    noisy = noisy[0:sampleLen]
    vec_CSIIh, vec_CSIIm, vec_CSIIl = fwseg_noise(clean, noisy, sample_rate)

    CSIIh = np.mean(vec_CSIIh)
    CSIIm = np.mean(vec_CSIIm)
    CSIIl = np.mean(vec_CSIIl)
    return CSIIh, CSIIm, CSIIl


def get_band(M, Fs):
    #   This function sets the bandpass filter band edges.
    # It assumes that the sampling frequency is 8000 Hz.
    A = 165
    a = 2.1
    K = 1
    L = 35
    CF = 300
    x_100 = (L / a) * np.log10(CF / A + K)
    CF = Fs / 2 - 600
    x_8000 = (L / a) * np.log10(CF / A + K)
    LX = x_8000 - x_100
    x_step = LX / M
    x = np.arange(x_100, x_8000 + x_step + 1e-20, x_step)
    if len(x) == M:
        np.append(x, x_8000)

    BAND = A * (10 ** (a * x / L) - K)
    return BAND


def get_ansis(BAND):
    fcenter = (BAND[0:-1] + BAND[1:]) / 2

    # Data from Table B.1 in "ANSI (1997). S3.5–1997 Methods for Calculation of the Speech Intelligibility
    # Index. New York: American National Standards Institute."
    f = np.array(
        [
            150,
            250,
            350,
            450,
            570,
            700,
            840,
            1000,
            1170,
            1370,
            1600,
            1850,
            2150,
            2500,
            2900,
            3400,
            4000,
            4800,
            5800,
            7000,
            8500,
        ]
    )
    BIF = np.array(
        [
            0.0192,
            0.0312,
            0.0926,
            0.1031,
            0.0735,
            0.0611,
            0.0495,
            0.0440,
            0.0440,
            0.0490,
            0.0486,
            0.0493,
            0.0490,
            0.0547,
            0.0555,
            0.0493,
            0.0359,
            0.0387,
            0.0256,
            0.0219,
            0.0043,
        ]
    )
    f_ANSI = interp1d(f, BIF)
    ANSIs = f_ANSI(fcenter)
    return fcenter, ANSIs


def ncm(clean, noisy, fs):
    if fs != 8000 and fs != 16000:
        raise ValueError("fs must be either 8 kHz or 16 kHz")

    x = clean  # clean signal
    y = noisy  # noisy signal
    F_SIGNAL = fs

    F_ENVELOPE = 32  # limits modulations to 0<f<16 Hz
    M_CHANNELS = 20

    #   DEFINE BAND EDGES
    BAND = get_band(M_CHANNELS, F_SIGNAL)

    #   Interpolate the ANSI weights in WEIGHT @ fcenter
    fcenter, WEIGHT = get_ansis(BAND)

    #   NORMALIZE LENGTHS
    Lx = len(x)
    Ly = len(y)

    if Lx > Ly:
        x = x[0:Ly]
    if Ly > Lx:
        y = y[0:Lx]

    Lx = len(x)
    Ly = len(y)

    X_BANDS = np.zeros((Lx, M_CHANNELS))
    Y_BANDS = np.zeros((Lx, M_CHANNELS))

    #   DESIGN BANDPASS FILTERS
    for a in range(M_CHANNELS):
        B_bp, A_bp = butter(
            4, np.array([BAND[a], BAND[a + 1]]) * (2 / F_SIGNAL), btype="bandpass"
        )
        X_BANDS[:, a] = lfilter(B_bp, A_bp, x)
        Y_BANDS[:, a] = lfilter(B_bp, A_bp, y)

    gcd = np.gcd(F_SIGNAL, F_ENVELOPE)
    #   CALCULATE HILBERT ENVELOPES, and resample at F_ENVELOPE Hz
    analytic_x = hilbert(X_BANDS, axis=0)
    X = np.abs(analytic_x)
    # X   = resample( X , round(len(x)/F_SIGNAL*F_ENVELOPE));
    X = resample_matlab_like(X, F_ENVELOPE, F_SIGNAL)
    analytic_y = hilbert(Y_BANDS, axis=0)
    Y = np.abs(analytic_y)
    # Y = resample( Y , round(len(x)/F_SIGNAL*F_ENVELOPE));
    Y = resample_matlab_like(Y, F_ENVELOPE, F_SIGNAL)
    ## ---compute weights based on clean signal's rms envelopes-----
    #
    Ldx, pp = X.shape
    p = 3  # power exponent - see Eq. 12

    ro2 = np.zeros((M_CHANNELS,))
    asnr = np.zeros((M_CHANNELS,))
    TI = np.zeros((M_CHANNELS,))

    for k in range(M_CHANNELS):
        x_tmp = X[:, k]
        y_tmp = Y[:, k]
        lambda_x = np.linalg.norm(x_tmp - np.mean(x_tmp)) ** 2
        lambda_y = np.linalg.norm(y_tmp - np.mean(y_tmp)) ** 2
        lambda_xy = np.sum((x_tmp - np.mean(x_tmp)) * (y_tmp - np.mean(y_tmp)))
        ro2[k] = (lambda_xy**2) / (lambda_x * lambda_y)
        asnr[k] = 10 * np.log10((ro2[k] + 1e-20) / (1 - ro2[k] + 1e-20))
        # Eq.9 in [1]

        if asnr[k] < -15:
            asnr[k] = -15
        elif asnr[k] > 15:
            asnr[k] = 15

        TI[k] = (asnr[k] + 15) / 30  # Eq.10 in [1]

    ncm_val = WEIGHT.dot(TI) / np.sum(WEIGHT)  # Eq.11
    return ncm_val
