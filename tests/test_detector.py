import os
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np

from detector import (
    ImageSourceDetector,
    combined_score,
    cosine_similarity,
    hash_similarity,
    phash_distance,
)


class ScoreTests(unittest.TestCase):
    def test_identical_vectors_have_cosine_one(self):
        vec = np.array([0.3, 0.4, 0.0], dtype=np.float32)
        self.assertAlmostEqual(cosine_similarity(vec, vec), 1.0, places=5)

    def test_orthogonal_vectors_have_cosine_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=5)

    def test_hash_similarity_is_one_when_distance_zero(self):
        self.assertEqual(hash_similarity(0), 1.0)
        self.assertAlmostEqual(hash_similarity(32), 0.5, places=5)

    def test_combined_score_weights_clip_higher(self):
        high_clip = combined_score(0.99, 10)
        high_hash = combined_score(0.50, 0)
        self.assertGreater(high_clip, high_hash)


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.detector = ImageSourceDetector(
            db_path=os.path.join(self.tmp.name, "net.pkl"),
            images_dir=os.path.join(self.tmp.name, "images"),
        )
        self.detector.add_user("bob", user_id="U_bob")
        self.detector.add_user("alice", user_id="U_alice")
        self.detector.follow("U_alice", "U_bob")

        original = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        near = np.array([0.98, 0.02, 0.0], dtype=np.float32)
        other = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        shared_hash = "ffffffffffffffff"
        later = datetime.now()
        earlier = later - timedelta(hours=2)

        self._add_fake_post("P_orig", "U_bob", original, shared_hash, earlier)
        self._add_fake_post("P_repost", "U_alice", near, shared_hash, later)
        self._add_fake_post("P_other", "U_alice", other, "0000000000000000", later)
        self.detector.graph.add_edge("P_repost", "P_orig", type="reposted_from")

    def tearDown(self):
        self.tmp.cleanup()

    def _add_fake_post(self, post_id, user_id, embedding, img_hash, timestamp):
        self.detector.graph.add_node(
            post_id,
            type="post",
            image_path="",
            caption="",
            timestamp=timestamp.isoformat(),
            likes=0,
        )
        self.detector.graph.add_edge(user_id, post_id, type="posted")
        self.detector.embeddings[post_id] = embedding
        self.detector.hashes[post_id] = img_hash

    def test_phash_distance_zero_for_same_hash(self):
        self.assertEqual(phash_distance("ffffffffffffffff", "ffffffffffffffff"), 0)

    def test_similar_search_finds_repost_and_skips_unrelated(self):
        matches = self.detector.find_similar(post_id="P_repost", clip_threshold=0.90, phash_threshold=5)
        ids = [m.post_id for m in matches]
        self.assertIn("P_orig", ids)
        self.assertNotIn("P_other", ids)

    def test_trace_walks_repost_edge_to_original_poster(self):
        result = self.detector.trace_source(post_id="P_repost")
        self.assertIsNotNone(result.origin)
        self.assertEqual(result.origin.post_id, "P_orig")
        self.assertEqual(result.origin.username, "bob")
        self.assertEqual(result.path, ["P_orig", "P_repost"])

    def test_original_post_is_its_own_source(self):
        result = self.detector.trace_source(post_id="P_orig")
        self.assertEqual(result.origin.post_id, "P_orig")
        self.assertEqual(result.path, ["P_orig"])

    def test_centrality_falls_back_to_follow_pagerank(self):
        source = self.detector.find_source_by_centrality(["P_orig", "P_repost"])
        self.assertEqual(source, "U_bob")


if __name__ == "__main__":
    unittest.main()
