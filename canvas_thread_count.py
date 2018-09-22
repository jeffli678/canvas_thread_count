import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import logging

def read_image(img_name):
    img = plt.imread(img_name).astype(float)
    logging.debug(img.shape)
    return img

def clear_log(log_name):
    with open(log_name, 'w'):
        pass

def calc_patch_density(img):
    # img is usually a 2cm * 2cm patch from the large paint

    rows, cols = img.shape
    crow, ccol = rows/2, cols/2

    # logging.info('doing fft')
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # logging.info('finding brightest point on the axis')
    max_x, _ = find_brightest_on_axis(fshift, crow, ccol, 0)
    max_y, _ = find_brightest_on_axis(fshift, crow, ccol, 1)

    # logging.info('calculating final result')
    f_x = ccol - max_x
    thread_x = f_x / float(px_to_cm(cols))
    f_y = crow - max_y
    thread_y = f_y / float(px_to_cm(rows))

    logging.debug('%d, %d' % (f_x, f_y))
    logging.info('%.2f, %.2f' % (thread_x, thread_y))
    return (fshift, thread_x, thread_y, crow, ccol, max_x, max_y)

def find_brightest_on_axis(img, crow, ccol, axis = 0):
    if axis == 0:
        # x axis
        max_x = 0
        # max_val = img[crow, 0]
        max_val = calc_average(img, crow, 4)
        
        for i in range(4, ccol - 20):
            aver = calc_average(img, crow, i)
            if aver > max_val:
                max_x = i
                max_val = aver

        return (max_x, max_val)

    else:
        # y axis
        max_y = 0
        max_val = calc_average(img, 4, ccol)
        
        for i in range(4, crow - 20):
            if calc_average(img, i, ccol) > max_val:
                max_y = i
                max_val = calc_average(img, i, ccol)

        return (max_y, max_val)       

def calc_average(img, row, col, extend = 1):

    patch = img[row - extend : row + extend, col - extend : col + extend]
    aver = np.average(patch)
    return aver

def px_to_cm(px, dpi = 600):
    return 2.54 * px / dpi

def cm_to_px(cm = 2, dpi = 600):
    return 600 * cm / 2.54

def show_magnitude(fshift, crow, ccol, max_x, max_y, annoted = True):

    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    _, ax = plt.subplots(1)
    plt.imshow(magnitude_spectrum, cmap = 'gray')

    if annoted:
        circ = Circle((max_x, crow),10)
        ax.add_patch(circ)
        circ = Circle((ccol, max_y),10)
        ax.add_patch(circ)

    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

def show_after_filter(fshift, crow, ccol, width_threshold = 4):

    fshift[0 : crow - width_threshold, 0 : ccol - width_threshold] = 0.01
    fshift[0 : crow - width_threshold, ccol + width_threshold :] = 0.01
    fshift[crow + width_threshold : , 0 : ccol - width_threshold] = 0.01
    fshift[crow + width_threshold : , ccol + width_threshold : ] = 0.01


    # fshift -= fshift
    # fshift[crow, max_x] = max_val_x
    # fshift[crow, cols - max_x] = max_val_x
    # fshift[max_y, ccol] = max_val_y
    # fshift[rows - max_y, ccol] = max_val_y


    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.imshow(img_back, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])

    plt.show()

def patch_generator(img, l = 2):

    # yield patches of size l cm * l cm

    px = int(cm_to_px(l))
    h, w = img.shape
    logging.info('h: %d, w: %d, patch: %d*%d' % (h, w, px, px))

    curr_h = 0
    curr_w = 0

    while curr_h + 120 < h:
        while curr_w < w:
            right = min(curr_w + px, w)
            bottom = min(curr_h + px, h)
            patch = img[curr_h : bottom, curr_w : right]
            yield patch

            curr_w += px

        curr_w = 0
        curr_h += px

def weigh_density(density_list, under_aver_threshold = 0.2, above_aver_threshold = 0.8):

    aver = np.average(density_list)
    logging.info('average: %.2f' % aver)

    # above average => red => (0, 1)
    # below average => blue => (-1, 0)

    sort_idx = np.argsort(density_list)
    l = len(density_list)

    density_tint = [idx * 2.0 / l - 1 for idx in sort_idx]    

    return density_tint


    # l = sorted(density_list)
    # min_threshold = l[int(len(l) * under_aver_threshold)]
    # max_threshold = l[int(len(l) * above_aver_threshold)]

    # density_res = []
    # for density in density_list:
    #     if density < min_threshold:
    #         density_res.append(-1)
    #     elif density > max_threshold:
    #         density_res.append(1)
    #     else:
    #         density_res.append(0)

    # return density_res

    # plt.hist(density_list)
    # plt.show()

def create_superimose(img, density_tint, patch_size, save_name, alpha = 0.5):
    
    # above average => red => (0, 1)
    # below average => blue => (-1, 0)

    h, w = img.shape
    h_n = int(h / patch_size)
    w_n = int(w / patch_size)
    idx = 0
    _, ax = plt.subplots(1)
    plt.imshow(img, cmap = 'gray')

    for i in range(h_n):
        for j in range(w_n + 1):
            
            c = density_tint[idx]
            if c >= 0:
                color = [c, 0, 0]
            else:
                color = [0, 0, -c]

            rect_left = j * patch_size
            rect_bottom = i * patch_size
            rect = Rectangle((rect_left, rect_bottom), patch_size, patch_size, alpha = alpha, color = color)
            ax.add_patch(rect)

            idx += 1 

    # plt.show()
    plt.savefig(save_name)
    print('image saved to %s' % save_name)



def main():
    
    log_name = 'log.txt'
    log_level = logging.INFO
    clear_log(log_name)
    logging.basicConfig(format = '%(message)s',\
                    filename = log_name, level = log_level)
    logging.info('=' * 50)

    # img = read_image('s0049V1962-4.portion.tif')

    img_list = ['s0049V1962-4.tif', 's0195V1962-3.tif']
    # img_list = ['s0195V1962-3.tif']

    for img_name in img_list:

        x_densities = []
        y_densities = []

        img = read_image('input/' + img_name)
        patch_gen = patch_generator(img)
        for patch in patch_gen:
            _, x_density, y_density, _, _, _, _  = calc_patch_density(patch)

            x_densities.append(x_density)
            y_densities.append(y_density)

        x_density_tint = weigh_density(x_densities)
        y_density_tint = weigh_density(y_densities)

        # print(x_density_tint)
        # print(y_density_tint)

        patch_size = cm_to_px(2)
        create_superimose(img, x_density_tint, patch_size, 'output/%s-%s.png' % (img_name, 'horizontal'))
        create_superimose(img, y_density_tint, patch_size, 'output/%s-%s.png' % (img_name, 'vertical'))

        print('done with %s' % img_name)

        # fshift, _, _, crow, ccol, max_x, max_y = calc_patch_density(img)
        # show_magnitude(fshift, crow, ccol, max_x, max_y)
        # show_after_filter(fshift, crow, ccol)


if __name__ == '__main__':
    main()