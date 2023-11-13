
#Display generated data
def image_check(gen_fake):
    img = gen_fake.data.numpy()
    for i in range(1):
        plt.imshow(img[i][0],cmap='gray')
        plt.show()
