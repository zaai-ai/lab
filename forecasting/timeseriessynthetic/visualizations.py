import matplotlib.pyplot as plt


def plot_augmentations(t, base, jittered, scaled, bootstrap):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axs[0].plot(t, base, label="Original", color="black")
    axs[0].set_title("Original Series")

    axs[1].plot(t, base, label="Original", color="black")
    axs[1].plot(t, jittered, label="Jittered", alpha=0.7)
    axs[1].plot(t, scaled, label="Scaled", alpha=0.7)
    axs[1].set_title("Parametric Transformations")

    axs[2].plot(t, base, label="Original", color="black")
    axs[2].plot(t[: len(bootstrap)], bootstrap, label="Block Bootstrap", color="orange")
    axs[2].set_title("Pattern‑Mixing: Block Bootstrap")

    for ax in axs:
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_advanced_mix(t, base, mixup, dba, stl_mbb_series):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(t, base, label="Original", color="black")
    axs[0].plot(t, mixup, label="TSMixup", color="purple")
    axs[0].set_title("Pattern Mixing: TSMixup")

    axs[1].plot(t, base, label="Original", color="black")
    axs[1].plot(t, dba, label="DBA", color="teal")
    axs[1].set_title("Pattern Mixing: DBA")

    axs[2].plot(t, base, label="Original", color="black")
    axs[2].plot(t, stl_mbb_series, label="STL + MBB", color="brown")
    axs[2].set_title("Decomposition + Bootstrapping (STL‑MBB)")

    for ax in axs:
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_generative_models(t, base, gmm, vae, gan):
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(t, base, label="Original", color="black")
    axs[0].set_title("Original Series")

    axs[1].plot(t, base, label="Original", color="black")
    axs[1].plot(t[: len(gmm)], gmm, label="GMM Samples", color="green")
    axs[1].set_title("Probabilistic Model (GMM)")

    axs[2].plot(t, base, label="Original", color="black")
    axs[2].plot(t, vae, label="VAE", color="blue")
    axs[2].set_title("Variational Auto‑Encoder")

    axs[3].plot(t, base, label="Original", color="black")
    axs[3].plot(t, gan, label="GAN", color="red")
    axs[3].set_title("Generative Adversarial Network")

    for ax in axs:
        ax.legend()

    plt.tight_layout()
    plt.show()
