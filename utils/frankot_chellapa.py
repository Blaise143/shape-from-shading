import numpy as np


def frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Integrate a gradient field (p, q) into a depth map z using
    the Frankot–Chellappa method.

    Args:
        p: ∂z/∂x (height x width)
        q: ∂z/∂y (height x width)
    Returns:
        z: reconstructed depth map (same shape)
    """
    h, w = p.shape
    wx = np.fft.fftfreq(w) * 2 * np.pi
    wy = np.fft.fftfreq(h) * 2 * np.pi
    WX, WY = np.meshgrid(wx, wy)

    Px = np.fft.fft2(p)
    Qy = np.fft.fft2(q)

    denom = WX**2 + WY**2
    denom[0, 0] = 1.0

    Z = (-1j * WX * Px - 1j * WY * Qy) / denom

    Z[0, 0] = 0.0

    z = np.real(np.fft.ifft2(Z))
    return z
