import torch
from torchvision import models, transforms
import numpy as np
import cv2
import time
from skimage.feature import peak_local_max
from model.EllipssianNet import EllipssianNet
import faiss
# Define the model
import math

class EllipssianNetManager:
    def __init__(self):
        self.model = None
        self.transform = None
        self.cov_scale_factor = 1
        self.width = None
        self.height = None
        self.depth_scale = 1.0

    def load_model(self, weight_path='model/EllipssianNet.pth'):
        self.model = EllipssianNet()
        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()  # Set to evaluation mode
        self.model = self.model.cuda()  # Move to GPU if available
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        try:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                dummy = torch.zeros(1, 3, 480, 640, device='cuda')
                _ = self.model(dummy)
                torch.cuda.synchronize()
        except Exception as e:
            print(f"[warn] warm-up failed: {e}")



    def inference(self, input_np):
        with torch.no_grad():  # No gradients needed for inference
            input_tensor = self.transform(input_np).unsqueeze(0).cuda()  # Add batch dimension and move to GPU
            gradient_output, center_output, cov_output = self.model(input_tensor)

            # Convert results back to CPU and detach from computation graph
            if gradient_output != None:
                gradient_output = gradient_output.squeeze()
            center_output = center_output.squeeze()
            cov_output = cov_output.squeeze()

        return gradient_output, center_output, cov_output


    def extract_centers(self, center_map):
        extracted_centers = peak_local_max(center_map*255, min_distance=5, threshold_abs=20)
        return extracted_centers

    def convert_cov2x2_denorm(self, cov_map):
        H = cov_map.shape[1]
        W = cov_map.shape[2]
        result_cov = torch.empty(2, 2, H, W)

        a_log = cov_map[0, :, :]
        b_norm = cov_map[1, :, :]
        c_log = cov_map[2, :, :]

        a = torch.expm1(a_log)
        c = torch.expm1(c_log)
        b = torch.sign(b_norm) * torch.expm1(b_norm.abs())

        result_cov[0, 0, :, :] = (a ** 2)
        result_cov[1, 1, :, :] = (b ** 2 + c ** 2)
        result_cov[0, 1, :, :] = (a * b)
        result_cov[1, 0, :, :] = (a * b)


        return result_cov


    def sample_cov_at_centers(self, extracted_centers, cov_map):
        # extracted_centers: N x 2 numpy
        # cov_map: 2 x 2 x W x H  # torch, cuda
        # rgb_img: H x W x 3  # numpy

        # Recover scale of covariance map
        y_coords, x_coords = extracted_centers[:, 0], extracted_centers[:, 1]
        extracted_covs = (cov_map[:, :, y_coords, x_coords]).permute(2, 0, 1)
        return   extracted_covs


    def filter_by_score(self, scores, extracted_centers, extracted_covs, std_score_threshold=1.0):
        # Convert std_scores to a NumPy array of std values
        std_scores = np.array([score['std'] for score in scores])
        mask = (std_scores < std_score_threshold)

        valid_indices = np.where(mask)[0]

        filtered_centers = extracted_centers[valid_indices]
        filtered_covs = extracted_covs[valid_indices]
        filtered_scores = [scores[i] for i in valid_indices]

        return filtered_scores, filtered_centers, filtered_covs


    def compute_scores_from_features_ellipses(self, centers, covariances, center_map):
        """
            Args
            ----
            centers       : (N, 2)  numpy  – (x, y)
            covariances   : (N, 2, 2) torch
            center_map    : (H, W)  numpy
            Returns
            -------
            List[dict] – [{'mse': float, 'std': float}, ...]  length: N
            """

        # ------------------------------------------------------------------
        # (0) Preparing
        # ------------------------------------------------------------------
        H, W = center_map.shape
        N = centers.shape[0]
        device = torch.device('cuda')
        cov_pd = (covariances / self.cov_scale_factor).to(device)  # (N,2,2)

        # ------------------------------------------------------------------
        # (1) Coordinate grid & diff
        # ------------------------------------------------------------------
        y_coords = torch.arange(H, dtype=torch.float32, device=device)  # (H,)
        x_coords = torch.arange(W, dtype=torch.float32, device=device)  # (W,)
        X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')  # (W,H)

        grid = torch.stack((X, Y), dim=-1)  # (W,H,2)
        grid = grid.unsqueeze(0).expand(N, -1, -1, -1)  # (N,W,H,2)

        centers_t = torch.as_tensor(centers[:, [1, 0]], dtype=torch.float32,
                                    device=device).view(N, 1, 1, 2)  # (N,1,1,2)

        diff = grid - centers_t  # (N,W,H,2)

        # ------------------------------------------------------------------
        # (2) Ellipse mask
        # ------------------------------------------------------------------
        Sigma_inv = torch.inverse(cov_pd)  # (N,2,2)

        # einsum: (N,W,H,2)*(N,2,2)*(N,W,H,2) → (N,W,H)
        d2 = torch.einsum('nwhk,nkl,nwhl->nwh', diff, Sigma_inv, diff)

        tau = 2.0  # 1 σ 경계. 2 σ 로 키우려면 tau = 4.0
        ellipse_mask = (d2 <= tau)  # (N,W,H) bool
        ellipse_mask = ellipse_mask.permute(0, 2, 1)  # (N,H,W)

        # ------------------------------------------------------------------
        # (3) center_map masking
        # ------------------------------------------------------------------
        center_map_t = torch.as_tensor(center_map, dtype=torch.float32,
                                       device=device).unsqueeze(0).repeat(N, 1, 1)  # (N,H,W)

        masked_center_map = torch.where(ellipse_mask,
                                        center_map_t,
                                        torch.zeros_like(center_map_t))  # (N,H,W)

        # ------------------------------------------------------------------
        # (4) Normalized center map
        # ------------------------------------------------------------------
        max_vals = masked_center_map.amax(dim=(1, 2), keepdim=True)  # (N,1,1)  ### CHANGED
        thr = 0.5
        valid = max_vals >= thr

        norm_center_map = torch.where(
            valid,
            masked_center_map / (max_vals + 1e-8),
            torch.zeros_like(masked_center_map)
        )

        # (5) Gaussian Hitmap -----------------------------
        # semi‑minor axis: sqrt(λ_min)
        eigen_vals, _ = torch.linalg.eigh(cov_pd)
        semi_minor = torch.sqrt(torch.clamp(eigen_vals[:, 0], min=1e-6))  # (N,)
        semi_minor = semi_minor.view(N, 1, 1)

        dist = torch.norm(diff, dim=-1).permute(0, 2, 1)  # (N,H,W)
        norm_dist = torch.clamp(dist / semi_minor, 0.0, 1.0)
        heat_map = 1 / (1 + 10 * norm_dist)  # (N,H,W)

        # ------------------------------------------------------------------
        # (6) Diff, MSE, STD
        # ------------------------------------------------------------------
        diff_map = (heat_map - norm_center_map)*100  # 1보다 작으므로, 100배 해줘야, 제곱했을 때 커짐.
        diff_map = torch.nan_to_num(diff_map, nan=0.0, posinf=0.0, neginf=0.0)

        std = (diff_map ** 2).std(dim=(1, 2))  # (N,)
        mean = (diff_map ** 2).mean(dim=(1, 2))  # (N,)


        return [{'mean': mean[i].item(),'std': std[i].item()} for i in range(N)]

    def non_max_suppression(self, centers, covariances, scores,
                                         overlap_threshold=0.5,
                                         shape_similarity_threshold=0.8):
        """
        Uses FAISS range_search() to prune overlapping ellipses.
        Each ellipse uses its major axis * overlap_threshold as the
        search radius (in Euclidean distance).

        Args:
            centers (ndarray): (N, 2), [y, x].
            covariances (torch.Tensor or ndarray): (N, 2, 2).
            scores (List[dict]): with 'mean' and 'std'.
            overlap_threshold (float): factor for the radius and overlap check.
            shape_similarity_threshold (float): similarity cutoff.

        Returns:
            List[int]: sorted list of kept ellipse indices.
        """
        N = len(centers)
        if N == 0:
            return []

        # Convert data to numpy if needed
        if hasattr(covariances, 'detach'):
            covariances = covariances.detach().cpu().numpy()  # shape (N, 2, 2)
        centers_np = np.array(centers, dtype=np.float32)      # for FAISS
        std_scores = np.array([s['std'] for s in scores], dtype=np.float32)

        # -- 1. Decompose to get ellipse parameters --
        eigenvalues, eigenvectors = np.linalg.eigh(covariances)
        eigenvalues = np.clip(eigenvalues, 1e-6, None)
        semi_axes = np.sqrt(eigenvalues)  # shape: (N, 2)
        semi_minor_axis = semi_axes[:, 0]
        semi_major_axis = semi_axes[:, 1]
        angles_deg = np.degrees(
            np.arctan2(eigenvectors[:, 1, 1], eigenvectors[:, 0, 1])
        )
        aspect_ratios = np.divide(
            semi_major_axis,
            semi_minor_axis,
            out=np.full_like(semi_major_axis, np.inf),
            where=(semi_minor_axis != 0)
        )

        # -- 2. Sort ellipses by ascending std --
        sorted_indices = np.argsort(std_scores)
        std_sorted = std_scores[sorted_indices]
        centers_sorted = centers_np[sorted_indices]
        semi_major_sorted = semi_major_axis[sorted_indices]
        semi_minor_sorted = semi_minor_axis[sorted_indices]
        aspect_ratios_sorted = aspect_ratios[sorted_indices]
        angles_sorted = angles_deg[sorted_indices]

        # -- 3. Build FAISS index (IndexFlatL2 for 2D) --
        index = faiss.IndexFlatL2(2)
        index.add(centers_sorted)  # shape: (N,2)

        keep = np.ones(N, dtype=bool)  # track which ellipses remain

        def shape_sim(ar1, angle1, ar2, angle2):
            """
            Compute shape similarity: a combo of aspect ratio + angle similarity.
            """
            eps = 1e-12
            ar_similarity = 1.0 - np.abs(ar1 - ar2) / (max(ar1, ar2) + eps)
            diff_angle = abs(angle1 - angle2)
            diff_angle = min(diff_angle, 180 - diff_angle)  # modulo 180
            angle_similarity = 1.0 - (diff_angle / 90.0)
            return np.clip(0.5 * (ar_similarity + angle_similarity), 0, 1)

        # -- 4. Range search per ellipse (in ascending std) --
        for i in range(N):
            if not keep[i]:
                continue  # already pruned

            center_i = centers_sorted[i].reshape(1, 2)
            maj_i = semi_major_sorted[i]
            min_i = semi_minor_sorted[i]
            if maj_i * min_i < 50:
                continue
            ar_i = aspect_ratios_sorted[i]
            angle_i = angles_sorted[i]

            # We'll use 'maj_i * overlap_threshold' as the Euclidean search radius
            # faiss.IndexFlatL2 uses L2-squared distances => (radius_sq)
            radius = min_i * (overlap_threshold + 1.0)
            radius_sq = radius * radius

            # -- Range search: returns (offsets, distances, labels) --
            #   offsets has shape (nq+1,) => for our single query, length=2
            #   distances, labels have length offsets[-1]
            offsets, dists, nbr_indices = index.range_search(center_i, radius_sq)

            # offsets[0] is always 0; offsets[1] is the total number of neighbors
            if len(offsets) < 2:
                # means no neighbors => continue
                continue

            start = offsets[0]
            end = offsets[1]
            # neighbor indices for ellipse i
            neighbors_j = nbr_indices[start:end]
            # distances_j = dists[start:end]  # if you want to check the actual L2-sq

            # -- Now filter neighbors by final overlap & shape check --
            for j in neighbors_j:
                if j == i or not keep[j]:
                    continue

                # Check overlap with sum of major axes => original logic
                dist_ij = np.linalg.norm(centers_sorted[i] - centers_sorted[j])
                sum_maj = maj_i + semi_major_sorted[j]
                if dist_ij <= overlap_threshold * sum_maj:
                    # shape similarity
                    sim = shape_sim(ar_i, angle_i,
                                    aspect_ratios_sorted[j], angles_sorted[j])
                    if sim >= shape_similarity_threshold:
                        # i has <= std than j => prune j
                        if std_sorted[j] >= std_sorted[i]:
                            keep[j] = False
                            # keep[j] = True

        # -- 5. Map back to original indices --
        kept_sorted = np.where(keep)[0]
        kept_original = sorted_indices[kept_sorted]

        valid_indices = sorted(kept_original.tolist())
        nms_centers = centers[valid_indices]
        nms_covs = covariances[valid_indices]
        nms_scores = [scores[i] for i in valid_indices]


        return nms_scores, nms_centers, torch.from_numpy(nms_covs), valid_indices



    def visualize_ellipses(self, centers, covariances, scores, center_map):
        # Normalize and convert the center map to RGB for visualization
        center_map_normalized = (center_map / np.max(center_map) * 255).astype(np.uint8)
        result_img = cv2.cvtColor(center_map_normalized, cv2.COLOR_GRAY2RGB)
        if hasattr(covariances, 'detach'):
            covariances = covariances.detach().cpu().numpy()  # shape (N, 2, 2)

        for i, (center, cov, score_dict) in enumerate(zip(centers, covariances, scores)):
            if score_dict['mean'] is None or score_dict['std'] is None:
                continue  # Skip if scores are not computed

            center_y, center_x = int(center[0]), int(center[1])

            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            semi_axes = np.sqrt(np.clip(eigenvalues, 1e-6, None))
            semi_major_axis = semi_axes[1]
            semi_minor_axis = semi_axes[0]
            angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

            # Draw ellipse
            cv2.ellipse(result_img, (center_x, center_y),
                        (int(semi_major_axis), int(semi_minor_axis)),
                        angle, 0, 360, (0, 255, 0), 2)

            # # Draw mean and std scores
            text_std = f"std: {score_dict['std']:.2f}"
            font_scale = 0.3
            font_thickness = 1
            size_m, baseline = cv2.getTextSize(text_std, cv2.FONT_HERSHEY_SIMPLEX,
                                           font_scale, font_thickness)
            text_x = center_x - size_m[0] // 2
            text_y = max(center_y - size_m[1] // 2, size_m[1])
            line_height = size_m[1] + baseline
            cv2.putText(result_img, text_std, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        font_thickness, cv2.LINE_AA)

        return result_img

    def visualize_color_ellipsses(self, center_list, cov_t, image):
        # Vectorized indexing: extract all covariance matrices for the given (y, x) points
        x_coords, y_coords = center_list[:, 1], center_list[:, 0]

        cov_np = cov_t.detach().cpu().numpy() if hasattr(cov_t, 'detach') else np.asarray(cov_t)
        colors = image[y_coords, x_coords, :]

        h, w, _ = image.shape
        result_img = np.full((h, w, 3), 255, dtype=np.uint8)

        # 2. (area, index) list
        areas_and_idx = []
        for i in range(cov_np.shape[0]):
            eigvals, _ = np.linalg.eigh(cov_np[i])
            eigvals = np.clip(eigvals, 0, None)  # Safety check for negatives
            a, b = np.sqrt(np.sort(eigvals)[::-1])  # a ≥ b
            area = a * b  # π는 생략해도 순서 동일
            areas_and_idx.append((area, i))

        # 3. Sort ellipses by size (bigger →  smaller)
        for _, i in sorted(areas_and_idx, key=lambda t: t[0], reverse=True):
            cov = cov_np[i]
            color = tuple(int(c) for c in colors[i])
            # if color == (255,255,255):
            #     continue

            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 0, None)
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            semi_major = np.sqrt(eigvals[0])
            semi_minor = np.sqrt(eigvals[1])
            angle_deg = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

            center = (int(center_list[i, 1]), int(center_list[i, 0]))

            # Contour(black) → fill(flat color)
            cv2.ellipse(result_img, center,
                        (int(semi_major), int(semi_minor)),
                        angle_deg, 0, 360, (50, 50, 50), 3)
            cv2.ellipse(result_img, center,
                        (int(semi_major), int(semi_minor)),
                        angle_deg, 0, 360, color, -1)

        return result_img


    def sample_color(self, center_list, image):
        # image: 480 x 640 fixed
        x_coords, y_coords = center_list[:, 1], center_list[:, 0]
        colors = image[y_coords, x_coords, :]
        return colors


    def recover_center_cov_with_size(self, center_list, cov_t, size_h_w):
        # Vectorized indexing: extract all covariance matrices for the given (y, x) points
        img_h = size_h_w[0]
        img_w = size_h_w[1]

        sx = 640 / img_w
        sy = 480 / img_h
        invsx = 1.0 / sx
        invsy = 1.0 / sy

        cov_np = cov_t.detach().cpu().numpy() if hasattr(cov_t, 'detach') else np.asarray(cov_t)

        # ---- VALUE correction only: Σ_orig = A^{-1} Σ_res A^{-T} ----
        # For diagonal A, elementwise:
        # Σ_xx /= sx^2,  Σ_yy /= sy^2,  Σ_xy /= (sx*sy)
        cov_np = cov_np.copy()
        cov_xx = cov_np[:, 0, 0] * (invsx * invsx)
        cov_yy = cov_np[:, 1, 1] * (invsy * invsy)
        cross = (invsx * invsy)
        cov_xy = cov_np[:, 0, 1] * cross
        cov_yx = cov_np[:, 1, 0] * cross
        cov_np[:, 0, 0] = cov_xx
        cov_np[:, 1, 1] = cov_yy
        cov_np[:, 0, 1] = cov_xy
        cov_np[:, 1, 0] = cov_yx  # keep symmetry if input is symmetric

        center_list_res = np.empty_like(center_list, dtype=np.float32)
        center_list_res[:, 0] = center_list[:, 0] * invsy  # y_res
        center_list_res[:, 1] = center_list[:, 1] * invsx  # x_res
        center_list_res_int = center_list_res.copy()
        center_list_res_int[:, 0] = np.clip(np.round(center_list_res_int[:, 0]).astype(int), 0, img_h - 1)
        center_list_res_int[:, 1] = np.clip(np.round(center_list_res_int[:, 1]).astype(int), 0, img_w - 1)

        return center_list_res, cov_np