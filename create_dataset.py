from argparse import ArgumentParser, Namespace
import sys

import os
import numpy as np
import cv2
from tqdm import tqdm
from model.dataset_creator import EllinetDatasetCreator

def convert_cov_cholesky_abc(cov):
    """Cholesky decomposition of 2x2xHxW covariance map → [3, H, W] (a, b, c)"""
    h, w = cov.shape[2], cov.shape[3]
    abc = np.zeros((3, h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            sigma = cov[:, :, i, j]  # shape [2, 2]
            try:
                L = np.linalg.cholesky(sigma)
                abc[0, i, j] = L[0, 0]  # a
                abc[1, i, j] = L[1, 0]  # b
                abc[2, i, j] = L[1, 1]  # c
            except np.linalg.LinAlgError:
                # Not PSD → fill with 0
                abc[:, i, j] = 0.0
    return abc


def normalize_log_cholesky(a, b, c):
    a_log = np.log1p(a)
    c_log = np.log1p(c)
    b_norm = np.sign(b) * np.log1p(np.abs(b))
    return a_log, b_norm, c_log


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--save_path', type=str, default="")
    parser.add_argument('--batch', type=str, default=100)
    parser.add_argument('--iteration', type=str, default=0)
    parser.add_argument('--begin_batch', type=str, default=0)
    parser.add_argument('--render', type=str, default="True")
    parser.add_argument('--optional_data', type=str, default="True")
    parser.add_argument('--edges', type=str, default="True")
    args = parser.parse_args(sys.argv[1:])

    batch = int(args.batch)
    begin_batch = int(args.begin_batch)

    # batch = 2
    tqdm_total = batch
    dataset_creator = EllinetDatasetCreator(width=640, height=480)



    # Save path
    save_path = args.save_path
    if not save_path == "":
        if not os.path.exists(save_path+"/center"):
            os.makedirs(save_path+"/center")
        if not os.path.exists(save_path + "/cov_raw"):
            os.makedirs(save_path + "/cov_raw")
        if not os.path.exists(save_path + "/cov_cholesky"):
            os.makedirs(save_path + "/cov_cholesky")
        if not os.path.exists(save_path + "/cov_cholesky_norm"):
            os.makedirs(save_path + "/cov_cholesky_norm")
        if not os.path.exists(save_path + "/gradient"):
            os.makedirs(save_path + "/gradient")
        if not os.path.exists(save_path + "/voronoi"):
            os.makedirs(save_path + "/voronoi")
        if not os.path.exists(save_path + "/ellipsses"):
            os.makedirs(save_path + "/ellipsses")
        if not os.path.exists(save_path + "/seeds"):
            os.makedirs(save_path + "/seeds")
        if not os.path.exists(save_path + "/edges"):
            os.makedirs(save_path + "/edges")

    print("save_path: ", save_path)

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
            idx = tqdm_total * int(args.iteration) + j

            # create a Set of data
            voronoi_img, voronoi_edges, voronoi_seeds, mean_t, gradient_map_np, center_map_np, cov_map_t \
                = dataset_creator.CreateSingleData()

            if render_switch or args.optional_data == "True":
                ellipse_img = dataset_creator.draw_ellipssians_on_image(mean_t, cov_map_t, voronoi_img)

                voronoi_seed_img = np.ones((dataset_creator.height, dataset_creator.width, 3),
                                           dtype=np.uint8) * 255  # White background
                for p in voronoi_seeds:
                    x, y = int(p[0]), int(p[1])
                    if 0 <= x < dataset_creator.width and 0 <= y < dataset_creator.height:
                        cv2.circle(voronoi_seed_img, (x, y), radius=2, color=(0, 0, 0), thickness=-1)

            if render_switch:
                cv2.imshow("Voronoi diagram", voronoi_img)
                cv2.imshow("Voronoi edges", voronoi_edges)
                cv2.imshow("Gradient_map", gradient_map_np)
                cv2.imshow("Ellipsses", ellipse_img)

                # Scale the transposed center_map to 0-255 for visualization
                center_map_scaled = (center_map_np * 255).astype(np.uint8)
                cv2.imshow("Center map", center_map_scaled)


                if len(voronoi_edges.shape) == 2 or voronoi_edges.shape[2] == 1:
                    voronoi_edges_color = cv2.cvtColor(voronoi_edges, cv2.COLOR_GRAY2BGR)
                else:
                    voronoi_edges_color = voronoi_edges
                combined_img = cv2.bitwise_and(voronoi_seed_img, voronoi_edges_color)
                cv2.imshow("Seeds", combined_img)
                cv2.waitKey(100)

            if not save_path == "":
                cv2.imwrite(save_path + f"/center/center_{idx:06d}.png", center_map_scaled)
                cov_map_np = cov_map_t.cpu().numpy()
                np.save(save_path + f"/cov_raw/cov_{idx:06d}.npy", cov_map_np)
                cov_cholesky = convert_cov_cholesky_abc(cov_map_np)
                np.save(save_path + f"/cov_cholesky/cov_{idx:06d}.npy", cov_cholesky)
                a_n, b_n, c_n = normalize_log_cholesky(cov_cholesky[0], cov_cholesky[1], cov_cholesky[2])
                cov_cholesky_norm = np.stack([a_n, b_n, c_n], axis=0)
                np.save(save_path+ f"/cov_cholesky_norm/cov_{idx:06d}.npy", cov_cholesky_norm)

                cv2.imwrite(save_path + f"/voronoi/voronoi_{idx:06d}.png", voronoi_img)
                cv2.imwrite(save_path + f"/gradient/gradient_{idx:06d}.png", np.uint8(gradient_map_np * 255))
                if args.optional_data == "True":
                    cv2.imwrite(save_path + f"/ellipsses/ellipsses_{idx:06d}.png", ellipse_img)
                    cv2.imwrite(save_path + f"/edges/edges_{idx:06d}.png", voronoi_edges)
                    cv2.imwrite(save_path + f"/seeds/seeds_{idx:06d}.png", voronoi_seed_img)

            j+=1
            pbar.update(1)



