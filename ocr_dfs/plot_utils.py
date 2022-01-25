from matplotlib import pyplot as plt
def print_only_one_img(image):
    plt.figure(num=None, figsize=(10, 10), facecolor='w', edgecolor='k')
    plt.title("Imagen original")
    plt.imshow(image)
    plt.axis("off")
    plt.show()