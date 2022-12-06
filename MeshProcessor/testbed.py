import os
from util.files import load_pcds, load_meshes
from proc.preprocessing import convert_img
from proc.clustering import get_largest_cluster, get_parts
import cv2

if __name__ == '__main__':
    dev_root = r'./data'
    #pcds = load_pcds(dev_root)
    pcds = load_meshes(dev_root)
    for name, pcd in pcds.items():
        pcd = get_largest_cluster(pcd)
        img_rgb = convert_img(pcd)
        img_bgr = img_rgb[..., ::-1]
        cv2.imwrite(os.path.join('./images', name + '.jpg'), img_bgr * 255)

        # img = cv2.imread(os.path.join('./images', name + '.jpg'), cv2.IMREAD_COLOR)
        # img = cv2.fastNlMeansDenoisingColored(img, None, 50, 50, 11, 15)
        # cv2.imwrite(os.path.join('./images', name + '.jpg'), img)

        get_parts(os.path.join('./images', name + '.jpg'), name)