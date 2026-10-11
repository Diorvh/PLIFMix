from __future__ import annotations

import unittest

from PLIFMix.ui.pages.distribution_page import display_name_suffix


class DisplayNameSuffixTests(unittest.TestCase):
    """The export name must say which view and which overlays were written."""

    def _suffix(
        self,
        *,
        image_mode: str = "ori_img",
        rotate_angle: int = 0,
        contour: bool = False,
        center_line: bool = False,
        fitted_center_line: bool = False,
    ) -> str:
        return display_name_suffix(
            image_mode=image_mode,
            rotate_angle=rotate_angle,
            show_contour=contour,
            show_center_line=center_line,
            show_fitted_center_line=fitted_center_line,
        )

    def test_plain_original_view_has_no_suffix(self) -> None:
        self.assertEqual(self._suffix(), "")

    def test_rotation_records_its_own_angle(self) -> None:
        self.assertEqual(self._suffix(image_mode="rot_img", rotate_angle=4), "-旋转4")
        self.assertEqual(self._suffix(image_mode="rot_img", rotate_angle=-1), "-旋转-1")
        self.assertEqual(self._suffix(image_mode="rot_img", rotate_angle=0), "-旋转0")

    def test_fit_and_delta_views_are_labelled(self) -> None:
        self.assertEqual(self._suffix(image_mode="fit_img"), "-拟合")
        self.assertEqual(self._suffix(image_mode="delta_img"), "-差值")

    def test_overlays_follow_the_view_in_a_stable_order(self) -> None:
        self.assertEqual(
            self._suffix(
                image_mode="rot_img",
                rotate_angle=6,
                contour=True,
                center_line=True,
                fitted_center_line=True,
            ),
            "-旋转6-轮廓-中心线-拟合中心线",
        )

    def test_view_is_written_before_overlays(self) -> None:
        self.assertEqual(self._suffix(image_mode="fit_img", fitted_center_line=True), "-拟合-拟合中心线")

    def test_overlays_alone_are_labelled(self) -> None:
        self.assertEqual(self._suffix(contour=True), "-轮廓")
        self.assertEqual(self._suffix(center_line=True), "-中心线")
        self.assertEqual(self._suffix(fitted_center_line=True), "-拟合中心线")


if __name__ == "__main__":
    unittest.main()
