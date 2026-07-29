import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from dataclasses import dataclass

from schema import ScarStateVector


class ScarFeatureExtractor:
    """
    Extracts a ScarStateVector from a photograph.

    Expects a cropped image of the scar region + surrounding skin.
    """

    def __init__(self, reference_scale_mm_per_px: float = None):
        """
        Args:
            reference_scale_mm_per_px:
                If a reference object (coin/ruler) was detected, pass the
                calibration here. Otherwise area will be in pixels, not mm².
        """
        self.scale = reference_scale_mm_per_px

    def extract(
        self,
        image_path: str,
        scar_mask: np.ndarray = None,
    ) -> ScarStateVector:
        """
        Main entry point.

        Args:
            image_path: Path to the scar photograph.
            scar_mask: Binary mask (same H×W as image) where
                1 = scar region.

                If None, automatic segmentation is attempted.

        Returns:
            ScarStateVector with all fields populated.
        """

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if scar_mask is None:
            scar_mask = self._segment_scar(img_lab)

        skin_mask = self._get_surrounding_skin_mask(
            scar_mask,
            dilation_px=40,
        )

        vascularity = self._compute_vascularity(
            img_lab,
            scar_mask,
        )

        pigmentation = self._compute_pigmentation(
            img_lab,
            scar_mask,
            skin_mask,
        )

        color_delta_e = self._compute_color_delta_e(
            img_lab,
            scar_mask,
            skin_mask,
        )

        texture_regularity = self._compute_texture_regularity(
            img_bgr,
            scar_mask,
        )

        surface_area = self._compute_surface_area(scar_mask)

        return ScarStateVector(
            vascularity=vascularity,
            pigmentation=pigmentation,
            pliability=0.0,  # Cannot be determined from a 2D image.
            height=0.0,      # Cannot be determined from a 2D image.
            surface_area_mm2=surface_area,
            texture_regularity=texture_regularity,
            color_delta_e=color_delta_e,
        )

    def _segment_scar(self, img_lab: np.ndarray) -> np.ndarray:
        """
        Basic scar segmentation using color deviation from surrounding skin.

        This is the naive baseline and will eventually be replaced by a
        learned segmentation model.
        """

        l_channel = img_lab[:, :, 0]
        a_channel = img_lab[:, :, 1]

        # Adaptive threshold on a* channel (red-green axis).
        blur = cv2.GaussianBlur(a_channel, (15, 15), 0)

        _, mask = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # Clean up using morphological operations.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        return (mask > 0).astype(np.uint8)

    def _get_surrounding_skin_mask(
        self,
        scar_mask: np.ndarray,
        dilation_px: int = 40,
    ) -> np.ndarray:
        """
        Get a ring of surrounding healthy skin.
        """

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_px, dilation_px),
        )

        dilated = cv2.dilate(
            scar_mask,
            kernel,
            iterations=1,
        )

        skin_mask = dilated - scar_mask
        return skin_mask

    def _compute_vascularity(
        self,
        img_lab: np.ndarray,
        scar_mask: np.ndarray,
    ) -> float:
        """
        Estimate vascularity using the a* channel in LAB color space.

        Returns a score on the Vancouver Scar Scale range (0–3).
        """

        a_channel = img_lab[:, :, 1].astype(np.float32)
        scar_pixels = a_channel[scar_mask == 1]

        if len(scar_pixels) == 0:
            return 0.0

        mean_a = np.mean(scar_pixels)

        # LAB a*:
        # 128 = neutral
        # >128 = increasingly red
        redness = max(0.0, mean_a - 128) / 128.0

        return min(3.0, redness * 3.0)

    def _compute_pigmentation(
        self,
        img_lab: np.ndarray,
        scar_mask: np.ndarray,
        skin_mask: np.ndarray,
    ) -> float:
        """
        Compare scar lightness against surrounding skin.

        Returns a normalized pigmentation abnormality score (0–3).
        """

        l_channel = img_lab[:, :, 0].astype(np.float32)

        scar_l = (
            np.mean(l_channel[scar_mask == 1])
            if np.any(scar_mask)
            else 128
        )

        skin_l = (
            np.mean(l_channel[skin_mask == 1])
            if np.any(skin_mask)
            else 128
        )

        diff = scar_l - skin_l

        # Larger difference = more abnormal pigmentation.
        abnormality = abs(diff) / 50.0

        return min(3.0, abnormality * 3.0)

    def _compute_color_delta_e(
        self,
        img_lab: np.ndarray,
        scar_mask: np.ndarray,
        skin_mask: np.ndarray,
    ) -> float:
        """
        Compute CIE76 Delta E between scar and surrounding skin.
        """

        img_float = img_lab.astype(np.float32)

        scar_mean = (
            np.mean(img_float[scar_mask == 1], axis=0)
            if np.any(scar_mask)
            else np.array([128, 128, 128])
        )

        skin_mean = (
            np.mean(img_float[skin_mask == 1], axis=0)
            if np.any(skin_mask)
            else np.array([128, 128, 128])
        )

        delta_e = np.sqrt(
            np.sum((scar_mean - skin_mean) ** 2)
        )

        return float(delta_e)

    def _compute_texture_regularity(
        self,
        img_bgr: np.ndarray,
        scar_mask: np.ndarray,
    ) -> float:
        """
        Estimate texture regularity using GLCM homogeneity.

        Returns:
            0 = irregular
            1 = smooth/regular
        """

        gray = cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        # Crop to scar bounding box.
        coords = np.where(scar_mask == 1)

        if len(coords[0]) == 0:
            return 1.0

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        scar_region = gray[
            y_min:y_max + 1,
            x_min:x_max + 1,
        ]

        if (
            scar_region.size == 0
            or scar_region.shape[0] < 2
            or scar_region.shape[1] < 2
        ):
            return 1.0

        # Quantize grayscale values to 16 levels.
        scar_quantized = (
            scar_region // 16
        ).astype(np.uint8)

        glcm = graycomatrix(
            scar_quantized,
            distances=[1],
            angles=[0],
            levels=16,
            symmetric=True,
            normed=True,
        )

        homogeneity = graycoprops(
            glcm,
            "homogeneity",
        )[0, 0]

        return float(homogeneity)

    def _compute_surface_area(
        self,
        scar_mask: np.ndarray,
    ) -> float:
        """
        Compute scar surface area.

        Returns:
            mm² if calibration is available,
            otherwise pixel count.
        """

        pixel_count = np.sum(scar_mask == 1)

        if self.scale is not None:
            return float(pixel_count * (self.scale ** 2))

        return float(pixel_count)