import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def generate(image: np.array) -> np.array:
    assert image.shape[0] == image.shape[1]

    npix = image.shape[0]

    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image) ** 2
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix // 2 + 1, 1.0)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    Abins, _, _ = scipy.stats.binned_statistic(
        knrm, fourier_amplitudes, statistic="mean", bins=kbins
    )
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

    plt.loglog(kvals, Abins)
    plt.xlabel("$k$")
    plt.ylabel("$P(k)$")
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    import cv2
    import bridson
    from rdfpy import rdf
    from scipy import fftpack
    from matplotlib.colors import LogNorm
    from tqdm import tqdm
    from scipy.interpolate import make_interp_spline, BSpline

    wh = 1000
    poisson_radius = 20.0

    im = np.ones([wh, wh]) * poisson_radius

    g_rl = []
    for i in tqdm(range(100)):
        num_samples, poisson_samples = bridson.poissonDiskSampling(im)

        poisson_samples /= wh

        # compute radial distribution function with step size = 0.1
        g_r, radii = rdf(poisson_samples, dr=0.01, parallel=False)
        g_rl.append(g_r)

    a = np.stack(g_rl).mean(axis=0)

    spl = make_interp_spline(radii, a, k=3)  # type: BSpline
    new_radii = np.linspace(radii.min(), radii.max(), 500)
    smoothed_values = spl(new_radii)

    x, y = np.meshgrid(new_radii - new_radii.max() / 2, new_radii - new_radii.max() / 2)
    dists = np.sqrt(x**2 + y**2)
    dists = spl(dists.flatten()).reshape(dists.shape)

    plt.imshow(dists, cmap="gray")
    plt.show()

    """
    point_image = np.zeros([wh, wh], dtype=np.uint8)

    for i in range(poisson_samples.shape[0]):
        point_image = cv2.circle(
            point_image, poisson_samples[i].astype(int), 3, color=1, thickness=-1
        )

    # point_image = cv2.circle(point_image, [500, 500], radius=250, color=1, thickness=10)

    im_fft = fftpack.fft2(point_image)
    im_fft = np.abs(np.fft.fftshift(im_fft))
    plt.imshow(im_fft, norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.figure()

    plt.imshow(point_image)
    plt.show()

    # ft = np.fft.ifftshift(point_image)
    ft = np.fft.fft2(point_image)
    ft = np.log(np.abs())

    std = ft.std()
    mean = ft.mean()
    factor = 10

    ft = np.where(ft > mean + factor * std, mean + factor * std, ft)

    plt.imshow(ft)
    plt.show(block=True)

    generate(point_image)
    """
