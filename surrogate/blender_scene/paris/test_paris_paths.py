from __future__ import annotations

import unittest

from paris_paths import (
    CODING_ROOT,
    GAIN_DIR,
    MESHES_DIR,
    ODB_DIR,
    OUTPUT_DATASET_DIR,
    PARIS_DIR,
    PARIS_SCENE_XML,
    PNG_DIR,
    SOURCE_DATASET_ROOT,
    SOURCE_TEST_RESULT_ROOT,
    SOURCE_TEST_ROOT,
)


class ParisPathsTest(unittest.TestCase):
    def test_core_paths_are_derived_from_current_workspace(self):
        self.assertEqual(PARIS_DIR.name, "paris")
        self.assertEqual(PARIS_DIR.parent.name, "blender_scene")
        self.assertEqual(CODING_ROOT.name, "coding")

    def test_output_paths_stay_under_current_paris_directory(self):
        self.assertEqual(OUTPUT_DATASET_DIR.parent, PARIS_DIR)
        self.assertEqual(PNG_DIR.parent, OUTPUT_DATASET_DIR)
        self.assertEqual(GAIN_DIR.parent, OUTPUT_DATASET_DIR)
        self.assertEqual(ODB_DIR.parent, PARIS_DIR)
        self.assertEqual(MESHES_DIR.parent, PARIS_DIR)
        self.assertEqual(PARIS_SCENE_XML.parent, PARIS_DIR)

    def test_reference_paths_follow_same_coding_root(self):
        self.assertEqual(SOURCE_TEST_ROOT.parent, CODING_ROOT)
        self.assertEqual(SOURCE_DATASET_ROOT.parent, SOURCE_TEST_ROOT)
        self.assertEqual(SOURCE_TEST_RESULT_ROOT.parent, SOURCE_TEST_ROOT)


if __name__ == "__main__":
    unittest.main()
