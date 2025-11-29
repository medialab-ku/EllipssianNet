import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from argparse import ArgumentParser
import sys

from PIL import Image
from model.manager import EllipssianNetManager

from skimage.feature import peak_local_max


def spatial_frequency_scalar(img: np.ndarray) -> float:
    """
    Compute the spatial frequency scalar using FFT.

    Args:
    img (np.ndarray): Input image array of shape (H, W, 3) or (H, W).
                      Values in [0, 1].

    Returns:
        float: Spatial frequency scalar (weighted average frequency).
    """
    # Convert to grayscale
    if img.ndim == 3 and img.shape[2] == 3:  # (H, W, 3)
        img_gray = img.mean(axis=2)
    else:  # already grayscale (H, W)
        img_gray = img.squeeze()

    # 2D FFT
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # frequency grid
    H, W = img_gray.shape
    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    freq_radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # weighted average frequency
    scalar = np.sum(freq_radius * magnitude) / np.sum(magnitude)
    return float(scalar)

if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--img_path', type=str, default="")
    args = parser.parse_args(sys.argv[1:])


    ellipssianNet_manager = EllipssianNetManager()
    ellipssianNet_manager.load_model(weight_path="./model/EllipssianNet.pth")


    # Define the transformation (ensure it matches what was used during training)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    img_path = args.img_path
    input_img = cv2.imread(img_path)
    img_w, img_h = input_img.shape[1], input_img.shape[0]

    # 0. Resize rgb to 480 x 640
    img_resize = cv2.resize(input_img , dsize=(640, 480))
    img_resize_blur = img_resize.copy()
    ttf = spatial_frequency_scalar(img_resize_blur)
    blur_list = [ttf]
    print("max:", img_resize_blur.max)
    print("TTF:", ttf)

    while ttf > 50.0:
        img_resize_blur = cv2.blur(img_resize_blur, (13, 13))
        ttf = spatial_frequency_scalar(img_resize_blur)
        blur_list.append(ttf)
        if len(blur_list) > 2:
            break

    gradient_result, center_result, cov_result = ellipssianNet_manager.inference(img_resize_blur)

    # Perfrom EllipssianNet
    gradient_recover = cv2.resize(gradient_result.detach().cpu().numpy(), dsize=(img_w, img_h))
    center_map = center_result.cpu().numpy()  # Numpy
    center_map_recover = cv2.resize(center_map.copy(), dsize=(img_w, img_h))

    ### Extract Centers by using Local Maxima (Peak)
    extracted_centers = ellipssianNet_manager.extract_centers(center_map)
    cov2x2_denorm = ellipssianNet_manager.convert_cov2x2_denorm(cov_result.detach())
    extracted_covs = ellipssianNet_manager.sample_cov_at_centers(extracted_centers, cov2x2_denorm)

    # Compute scores for filtering
    scores = ellipssianNet_manager.compute_scores_from_features_ellipses(extracted_centers, extracted_covs, center_map)

    # Score-Based Filtering
    # Large std_score_threshold, Many ellipses survive
    filtered_scores, filtered_centers, filtered_covs \
        = ellipssianNet_manager.filter_by_score(scores, extracted_centers, extracted_covs, std_score_threshold=80)

    # Non-Maximum Suppression (Filter the overlapping Ellipses based on scores)
    # Large overlap_threshold, Many ellipses get checked (the possibility of getting filtered increases)
    # Large shape_similarity_threshold, Many ellipses survives
    nms_scores, nms_centers, nms_covs, valid_indices \
        = ellipssianNet_manager.non_max_suppression(filtered_centers, filtered_covs, filtered_scores,
                                              overlap_threshold=0.5, shape_similarity_threshold=0.2)
    sampled_colors = ellipssianNet_manager.sample_color(nms_centers, img_resize.copy())
    center_recovered_size, cov_recovered_size = ellipssianNet_manager.recover_center_cov_with_size(nms_centers, nms_covs,
                                                                                             (img_h, img_w))

    ## Visualize Begins
    filtered_ellipses = ellipssianNet_manager.visualize_ellipses(nms_centers, nms_covs,
                                                           nms_scores, center_map)
    initial_ellipses = ellipssianNet_manager.visualize_ellipses(extracted_centers, extracted_covs,
                                                          scores, center_map)

    filtered_ellipses_recover = cv2.resize(filtered_ellipses.copy(), dsize=(img_w, img_h))
    initial_ellipses_recover = cv2.resize(initial_ellipses.copy(), dsize=(img_w, img_h))
    # Display the result image
    cv2.imshow('Filtered Ellipses', filtered_ellipses_recover)
    cv2.imshow('Initial Ellipses', initial_ellipses_recover)

    colored_ellipses = ellipssianNet_manager.visualize_color_ellipsses(nms_centers, nms_covs, img_resize.copy())
    colored_ellipses_recover = cv2.resize(colored_ellipses.copy(), dsize=(img_w, img_h))
    cv2.imshow("Color Ellipses", colored_ellipses_recover)


    cv2.imshow("Voronoi diagram", input_img)
    cv2.imshow("Voronoi gradient_map", gradient_result.cpu().numpy())
    cv2.imshow("Center probability map", center_result.cpu().numpy())
    cv2.waitKey(0)




