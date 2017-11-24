import matplotlib.pyplot as plt


def show_images(imgs, n_pixel, m_pixel, show_inarow=10):
    n_imgs = len(imgs)
    
    plt_n = (n_imgs - 1) / show_inarow  + 1
    plt_m = show_inarow
    plt.figure(figsize=(2 * plt_m, 2 * plt_n))
    
    for i, img in enumerate(imgs):
        to_draw = img.reshape(n_pixel, m_pixel)
        ax = plt.subplot(plt_n, plt_m, i + 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.gray()
        plt.imshow(to_draw)
    plt.show()
