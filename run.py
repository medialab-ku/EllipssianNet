import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from argparse import ArgumentParser
import sys

from PIL import Image
from EllipssianNetCNN import EllipssianNetCNN

from skimage.feature import peak_local_max


# Function to perform inference
def inference(image_path):
    # Load and preprocess the input image
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).cuda()  # Add batch dimension and move to GPU

    # Perform inference
    with torch.no_grad():  # No gradients needed for inference
        gradient_output, center_output, cov_output = model(input_tensor, training=True)

    # Convert results back to CPU and detach from computation graph
    gradient_output = gradient_output.squeeze().cpu().numpy()
    center_output = center_output.squeeze().cpu().numpy()
    cov_output = cov_output.squeeze().cpu().numpy()

    return gradient_output, center_output, cov_output



def draw_ellipssians_on_image(centers, cov_map, image):
    # Convert means and covariances to numpy for easier manipulation

    H_factor = image.shape[0] * 0.6
    W_factor = image.shape[1] * 0.6

    cov_map[0, 0, :, :] *= (W_factor ** 2)  # Normalize the variance along x-axis
    cov_map[1, 1, :, :] *= (H_factor ** 2)  # Normalize the variance along y-axis
    cov_map[0, 1, :, :] *= (W_factor * H_factor)  # Normalize the covariance between x and y
    cov_map[1, 0, :, :] *= (W_factor * H_factor)  # Normalize the covariance between x and y (symmetric entry)

    # Vectorized indexing: extract all covariance matrices for the given (y, x) points
    x_coords, y_coords = centers[:, 1], centers[:, 0]

    # Extract covariances using advanced indexing
    extracted_covariances = cov_map[:, :, x_coords, y_coords]
    extracted_colors = image[y_coords, x_coords, :]

    result_img = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)
    result_img.fill(255)  # or img[:] = 255


    for i in range(extracted_colors.shape[0]):
        cov = extracted_covariances[:, :, i]
        color = (int(extracted_colors[i, 0]), int(extracted_colors[i, 1]), int(extracted_colors[i, 2]))


        # Perform eigenvalue decomposition to get the axes lengths and orientation
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.clip(eigenvalues, 0, None)

        # Sort eigenvalues and eigenvectors (to ensure correct assignment to major/minor axis)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]



        # Compute the semi-major and semi-minor axes (as 2 standard deviations for visualization)
        semi_major_axis = np.sqrt(eigenvalues[0])
        semi_minor_axis = np.sqrt(eigenvalues[1])

        # Compute the angle of the ellipse (in degrees)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        # Convert the mean to int for OpenCV drawing
        center = (int(centers[i, 1]), int(centers[i, 0]))

        # Draw the ellipse using OpenCV
        cv2.ellipse(result_img, center, (int(semi_major_axis), int(semi_minor_axis)), angle, 0, 360, (0, 0, 0), 3)
        cv2.ellipse(result_img, center, (int(semi_major_axis), int(semi_minor_axis)), angle, 0, 360, color, -1)

    return result_img



if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--img_path', type=str, default="")
    args = parser.parse_args(sys.argv[1:])

    model = EllipssianNetCNN()
    model.load_state_dict(torch.load('E:/ISMAR_2025/weightEllipssianNetCNN.pth'))
    model.eval()  # Set to evaluation mode
    model = model.cuda()  # Move to GPU if available

    # Define the transformation (ensure it matches what was used during training)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    img_path = args.img_path
    input_img = cv2.imread(img_path)
    gradient_result, center_result, cov_result = inference(img_path)

    # Center points are extracted with local maxima
    coordinates = peak_local_max(center_result*256, min_distance=5, threshold_abs=30)
    center_point_image = cv2.cvtColor(center_result, cv2.COLOR_GRAY2BGR)
    for coord in coordinates:
        cv2.circle(center_point_image, (coord[1], coord[0]), 3, (0, 0, 255), -1)


    ellipssian_img = draw_ellipssians_on_image(coordinates, cov_result, input_img.copy())

    cv2.imshow("Voronoi diagram", input_img)
    cv2.imshow("Voronoi gradient_map", gradient_result)
    cv2.imshow("Center probability map", center_result)
    cv2.imshow("Center points map", center_point_image)
    cv2.imshow("Ellipssian", ellipssian_img)
    cv2.waitKey(0)




