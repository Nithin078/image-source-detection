import os
import tempfile
import unittest
from datetime import datetime, timedelta

import networkx as nx
import numpy as np

from detector import (
    ImageSourceDetector,
    combined_score,
    cosine_similarity,
    hash_similarity,
    image_timestamp,
    is_link_match,
    is_search_match,
    parse_exif_datetime,
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

    def test_phash_distance_is_plain_int(self):
        dist = phash_distance("ffffffffffffffff", "ffffffffffffffff")
        self.assertIsInstance(dist, int)
        self.assertNotIsInstance(dist, np.integer)
        self.assertEqual(dist, 0)

    def test_search_is_looser_than_link(self):
        self.assertTrue(is_search_match(0.86, 10))
        self.assertFalse(is_link_match(0.86, 10))
        self.assertTrue(is_link_match(0.95, 2))
        self.assertTrue(is_link_match(0.92, 0))


class GraphFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.detector = ImageSourceDetector(data_dir=self.tmp.name)
        self.detector.add_user("bob", user_id="U_bob")
        self.detector.add_user("alice", user_id="U_alice")
        self.detector.add_user("carol", user_id="U_carol")
        self.detector.follow("U_alice", "U_bob")

        original = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        near = np.array([0.98, 0.02, 0.0], dtype=np.float32)
        other = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        shared_hash = "ffffffffffffffff"
        later = datetime.now()
        earlier = later - timedelta(hours=2)

        self._add_fake_post("P_orig", "U_bob", original, shared_hash, earlier)
        self._add_fake_post("P_repost", "U_alice", near, shared_hash, later)
        self._add_fake_post("P_other", "U_carol", other, "0000000000000000", later)

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


class TraceTests(GraphFixture):
    def test_similar_search_finds_repost_and_skips_unrelated(self):
        matches = self.detector.find_similar(post_id="P_repost", clip_threshold=0.85, phash_threshold=5)
        ids = [m.post_id for m in matches]
        self.assertIn("P_orig", ids)
        self.assertNotIn("P_other", ids)

    def test_trace_walks_repost_edge_to_original_poster(self):
        self.detector.graph.add_edge("P_repost", "P_orig", type="reposted_from")
        result = self.detector.trace_source(post_id="P_repost")
        self.assertIsNotNone(result.origin)
        self.assertEqual(result.origin.post_id, "P_orig")
        self.assertEqual(result.origin.username, "bob")
        self.assertEqual(result.path, ["P_orig", "P_repost"])
        self.assertFalse(result.used_edge_fallback)

    def test_trace_falls_back_to_earliest_match_without_repost_edge(self):
        result = self.detector.trace_source(post_id="P_repost")
        self.assertEqual(result.origin.post_id, "P_orig")
        self.assertEqual(result.origin.username, "bob")
        self.assertTrue(result.used_edge_fallback)
        self.assertEqual(result.path[0], "P_orig")

    def test_original_post_is_its_own_source(self):
        result = self.detector.trace_source(post_id="P_orig")
        self.assertEqual(result.origin.post_id, "P_orig")
        self.assertEqual(result.path, ["P_orig"])

    def test_origin_prefers_earliest_match_over_pagerank(self):
        self.detector.follow("U_carol", "U_alice")
        result = self.detector.trace_source(post_id="P_repost")
        amplifier = self.detector.find_source_by_centrality(["P_orig", "P_repost"])
        self.assertEqual(amplifier, "U_bob")
        self.assertEqual(result.origin.post_id, "P_orig")
        self.assertEqual(result.centrality_username, "bob")

        later_viral = np.array([0.97, 0.03, 0.0], dtype=np.float32)
        self._add_fake_post(
            "P_viral",
            "U_alice",
            later_viral,
            "ffffffffffffffff",
            datetime.now(),
        )
        self.assertEqual(self.detector.find_source_by_centrality(["P_orig", "P_viral"]), "U_bob")
        origin = self.detector.trace_source(post_id="P_viral").origin
        self.assertEqual(origin.post_id, "P_orig")

    def test_author_can_like_own_post_without_losing_posted_edge(self):
        self.assertIsInstance(self.detector.graph, nx.MultiDiGraph)
        likes = self.detector.like("U_bob", "P_orig")
        self.assertEqual(likes, 1)
        self.assertEqual(self.detector.post_author("P_orig"), "U_bob")
        self.assertTrue(self.detector._has_typed_edge("U_bob", "P_orig", "posted"))
        self.assertTrue(self.detector._has_typed_edge("U_bob", "P_orig", "likes"))


class PersistenceTests(GraphFixture):
    def test_json_and_npy_roundtrip(self):
        self.detector.save()
        self.assertTrue(os.path.isfile(os.path.join(self.tmp.name, "graph.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.tmp.name, "hashes.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.tmp.name, "embeddings", "P_orig.npy")))

        reloaded = ImageSourceDetector(data_dir=self.tmp.name)
        self.assertIn("U_bob", reloaded.graph)
        self.assertIn("P_orig", reloaded.embeddings)
        self.assertEqual(reloaded.hashes["P_orig"], "ffffffffffffffff")
        self.assertEqual(reloaded.post_author("P_orig"), "U_bob")
        np.testing.assert_allclose(reloaded.embeddings["P_orig"], self.detector.embeddings["P_orig"])


class SampleImageTests(unittest.TestCase):
    def test_sample_images_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            detector = ImageSourceDetector(data_dir=tmp)
            paths = detector.create_sample_images()
            self.assertTrue(os.path.isfile(paths["original"]))
            self.assertTrue(os.path.isfile(paths["near"]))
            self.assertTrue(os.path.isfile(paths["other"]))


class TimestampTests(unittest.TestCase):
    def test_parse_exif_datetime(self):
        parsed = parse_exif_datetime("2024:06:13 11:47:38")
        self.assertEqual(parsed, datetime(2024, 6, 13, 11, 47, 38))
        self.assertIsNone(parse_exif_datetime("not-a-date"))

    def test_generated_image_falls_back_to_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            detector = ImageSourceDetector(data_dir=tmp)
            path = detector.create_sample_images()["original"]
            expected = datetime.fromtimestamp(os.path.getmtime(path))
            stamp, source = image_timestamp(path)
            self.assertEqual(source, "mtime")
            self.assertAlmostEqual(stamp.timestamp(), expected.timestamp(), delta=2)


class DeleteResetTests(GraphFixture):
    def test_delete_post_removes_fingerprint_and_keeps_author(self):
        self.detector.delete_post("P_other")
        self.assertNotIn("P_other", self.detector.graph)
        self.assertNotIn("P_other", self.detector.embeddings)
        self.assertNotIn("P_other", self.detector.hashes)
        self.assertIn("U_carol", self.detector.graph)

    def test_delete_user_removes_their_posts(self):
        removed = self.detector.delete_user("U_alice")
        self.assertEqual(removed, 1)
        self.assertNotIn("U_alice", self.detector.graph)
        self.assertNotIn("P_repost", self.detector.graph)
        self.assertIn("P_orig", self.detector.graph)

    def test_unfollow_and_unlike(self):
        self.detector.like("U_bob", "P_orig")
        self.assertEqual(self.detector.unlike("U_bob", "P_orig"), 0)
        self.detector.unfollow("U_alice", "U_bob")
        self.assertEqual(self.detector.follows(), [])

    def test_reset_clears_graph_and_store(self):
        self.detector.save()
        self.detector.reset()
        self.assertEqual(list(self.detector.graph.nodes()), [])
        self.assertEqual(self.detector.embeddings, {})
        self.assertEqual(self.detector.hashes, {})
        reloaded = ImageSourceDetector(data_dir=self.tmp.name)
        self.assertEqual(list(reloaded.graph.nodes()), [])


if __name__ == "__main__":
    unittest.main()
