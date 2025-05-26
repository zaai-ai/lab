import random
import numpy as np
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.seasonal import STL


def parametric_transformations(base: np.ndarray):
    jittered = base + 0.5 * np.random.randn(len(base))
    scaled = 2 * base
    return jittered, scaled


def block_bootstrap(series: np.ndarray, block_size: int = 50):
    blocks = [
        series[i : i + block_size]
        for i in range(0, len(series) - block_size, block_size)
    ]
    random.shuffle(blocks)
    return np.concatenate(blocks[: len(series) // block_size])


def ts_mixup(a: np.ndarray, b: np.ndarray, alpha: float = 0.4):
    lam = np.random.beta(alpha, alpha)
    return lam * a + (1 - lam) * b


def gmm_sampling(series: np.ndarray, segment_size: int = 25):
    segments = np.array(
        [series[i : i + segment_size] for i in range(0, len(series) - segment_size + 1)]
    )
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
    gmm.fit(segments)
    samples, _ = gmm.sample(len(series) // segment_size)
    return np.concatenate(samples), segments.astype(np.float32)


def stl_mbb(series: np.ndarray, period: int = 50, block: int = 50):
    def moving_block_bootstrap(residuals: np.ndarray, block: int):
        n = len(residuals)
        resampled = [
            residuals[np.random.randint(0, n - block) :][:block]
            for _ in range(int(np.ceil(n / block)))
        ]
        return np.concatenate(resampled)[:n]

    stl = STL(series, period=period)
    result = stl.fit()
    boot_resid = moving_block_bootstrap(result.resid, block)
    return result.trend + result.seasonal + boot_resid
