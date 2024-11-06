from argparse import ArgumentParser, Namespace
import sys

from voronoi import Voronoi
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ellipssian_computer import EllipssianComputer


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--save_path', type=str, default="")
    parser.add_argument('--batch', type=str, default=100)
    parser.add_argument('--iteration', type=str, default=0)
    parser.add_argument('--begin_batch', type=str, default=0)
    parser.add_argument('--render', type=str, default="True")
    parser.add_argument('--ellipssian', type=str, default="False")
    parser.add_argument('--edges', type=str, default="False")
    args = parser.parse_args(sys.argv[1:])

    batch = int(args.batch)
    begin_batch = int(args.begin_batch)

    # batch = 2
    tqdm_total = batch
    width = 640
    height = 480
    voronoi = Voronoi(width=width, height=height)


    white_img = np.zeros([height, width, 3], dtype=np.uint8)
    white_img.fill(255)  # or img[:] = 255
    black_img = cv2.cvtColor(white_img.copy(), cv2.COLOR_BGR2GRAY)
    black_img.fill(0)


    # Save path
    save_path = args.save_path
    if not save_path == "":
        if not os.path.exists(save_path+"/center"):
            os.makedirs(save_path+"/center")
        if not os.path.exists(save_path + "/cov"):
            os.makedirs(save_path + "/cov")
        if not os.path.exists(save_path + "/gradient"):
            os.makedirs(save_path + "/gradient")
        if not os.path.exists(save_path + "/voronoi"):
            os.makedirs(save_path + "/voronoi")
        if not os.path.exists(save_path + "/ellipssian"):
            os.makedirs(save_path + "/ellipssian")

    if args.render == "False":
        render_switch = False
    else:
        render_switch = True


    with tqdm(total=tqdm_total) as pbar:
        j = 0
        while j < begin_batch:
            j+=1
            pbar.update(1)
        while j < tqdm_total:
            ellipssian_computer = EllipssianComputer(width=width, height=height)
            idx = tqdm_total * int(args.iteration) + j
            voronoi.SetParams()
            seeds = voronoi.generate_random_points()
            polygon_list, color_list, img, lined_img, gradient_map, polygon_mask_list, edge_mask_list = voronoi.ComputeVoronoi(seeds)

            identity_matrix = torch.eye(2).unsqueeze(-1).unsqueeze(-1).float().cuda()  # Shape: (2, 2, 1, 1)
            init_cov_map = identity_matrix.repeat(1, 1, img.shape[1], img.shape[0])  # Shape: (2, 2, N, M)
            mean_t, cov_map, center_map = ellipssian_computer.CreateMaps(init_cov_map, polygon_mask_list, edge_mask_list)
            cov_map_np = cov_map.cpu().numpy()
            # cov_map_swapped = cov_map.cpu().numpy().transpose(2, 3, 0, 1).reshape(480, 640, 4)
            center_map_swapped = center_map.cpu().numpy().T  # Transpose the array to swap dimensions
            # Normalize the transposed center_map to 0-255 for visualization
            center_map_normalized = (center_map_swapped * 255).astype(np.uint8)

            if render_switch or args.ellipssian == "True":
                ellipssian_img = ellipssian_computer.draw_ellipssians_on_image(mean_t, cov_map, img)

            if render_switch:
                cv2.imshow("Voronoi diagram", img)
                cv2.imshow("Voronoi edges", lined_img)
                cv2.imshow("Voronoi gradient_map", gradient_map)
                cv2.imshow("Center probability map", center_map_normalized)
                cv2.imshow("Ellipssian", ellipssian_img)
                cv2.waitKey(1)

            if not save_path == "":
                cv2.imwrite(save_path + f"/center/center_{idx:06d}.png", center_map_normalized)
                np.save(save_path + f"/cov/cov_{idx:06d}.npy", cov_map_np)
                cv2.imwrite(save_path + f"/voronoi/voronoi_{idx:06d}.png", img)
                cv2.imwrite(save_path + f"/gradient/gradient_{idx:06d}.png", np.uint8(gradient_map * 255))
                if args.ellipssian == "True":
                    cv2.imwrite(save_path + f"/ellipssian/ellipssian_{idx:06d}.png", ellipssian_img)
                if args.edges == "True":
                    cv2.imwrite(save_path + f"/edges/edges_{idx:06d}.png", lined_img)
            j+=1
            pbar.update(1)



