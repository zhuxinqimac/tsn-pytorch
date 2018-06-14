import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc

# w=10
# h=10
# fig=plt.figure(figsize=(8, 8))
# columns = 4
# rows = 5
# for i in range(1, columns*rows +1):
    # img = np.random.randint(10, size=(h,w))
    # fig.add_subplot(rows, columns, i)
    # plt.imshow(img)
# # plt.show()
# plt.savefig('./imgs/test.pdf', dpi=300)


def show_frames_att(imgs, atts, video_names, img_name='./imgs/default.pdf'):
    imgs = np.array(imgs)
    atts = np.array(atts)
    imgs = imgs*atts
    # imgs = (imgs*atts).astype(int)
    fig = plt.figure(figsize=(8, 8))
    print(imgs.shape)
    rows, columns = imgs.shape[0], imgs.shape[1]
    for i in range(1, columns*rows+1):
        img = imgs[(i-1)//columns, (i-1)%columns, ...]
        att = atts[(i-1)//columns, (i-1)%columns, ...]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
        if (i-1)%columns == 0:
            plt.title(video_names[(i-1)//columns], loc='left', fontsize=10)
    plt.savefig(img_name, dpi=200)

# imgs_each = []
# for i in range(3):
    # imgs_each.append(
            # mpimg.imread('/home/xinqizhu/Something_frames/72594/0000%d.jpg'%(i+1)))
# imgs = []
# imgs.append(imgs_each)
# imgs.append(imgs_each)
# imgs.append(imgs_each)
# imgs.append(imgs_each)
# # atts = np.ones([4,3,100,132,3])
# atts = np.ones([4,3,7,7,3])
# atts[0, 0, 1, 1, ...] = 0.8
# atts[0, 1, 5, 5, ...] = 0.5
# atts[0, 2, 5, 5, ...] = 0.5
# # atts[1, ...] = 0.2
# # print(atts)
# atts_resized = []
# # print(atts.shape)
# for i in range(atts.shape[0]):
    # tmp = []
    # for j in range(atts.shape[1]):
        # tmp.append(scipy.misc.imresize(atts[i,j], (100, 132)))
    # # print(tmp)
    # atts_resized.append(tmp)
# atts_resized = np.array(atts_resized)/256.
# # atts_resized = np.expand_dims(atts_resized, axis=-1)
# # atts_resized = np.repeat(atts_resized, 3, axis=-1)
# # atts_resized.fill(1.)
# # print(np.max(atts_resized))
# # print(np.min(atts_resized))
# # print(atts_resized[1,0,...])
# # imgs = (imgs*atts_resized).astype(int)
# show_frames_att(imgs, atts_resized)
