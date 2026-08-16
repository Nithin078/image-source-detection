"""
Image source detection: CLIP embeddings + perceptual hash + NetworkX graph.

The social network is a directed graph of users and posts. When a new image
is posted, it is compared against stored fingerprints. Near-matches are linked
with a reposted_from edge. Source tracing walks those edges and ranks
candidates by visual similarity and timestamp.
"""

from __future__ import annotations

import os
import pickle
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import imagehash
import networkx as nx
import numpy as np
from PIL import Image

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_CLIP_THRESHOLD = 0.90
DEFAULT_PHASH_THRESHOLD = 5
PHASH_BITS = 64


def _now() -> datetime:
    return datetime.now()


def _parse_ts(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return _now()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def phash_distance(hash_a: str, hash_b: str) -> int:
    return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)


def hash_similarity(distance: int) -> float:
    return max(0.0, 1.0 - (distance / PHASH_BITS))


def combined_score(clip_sim: float, phash_dist: int) -> float:
    return 0.7 * clip_sim + 0.3 * hash_similarity(phash_dist)


@dataclass
class Match:
    post_id: str
    user_id: str
    username: str
    clip_similarity: float
    phash_distance: int
    combined: float
    timestamp: datetime
    image_path: str
    caption: str = ""
    is_origin: bool = False


@dataclass
class TraceResult:
    query_post_id: Optional[str]
    origin: Optional[Match]
    matches: List[Match] = field(default_factory=list)
    path: List[str] = field(default_factory=list)
    reasoning: str = ""
    centrality_user_id: Optional[str] = None
    centrality_username: Optional[str] = None


class ImageSourceDetector:
    def __init__(
        self,
        db_path: str = os.path.join("data", "image_network.pkl"),
        images_dir: str = os.path.join("data", "stored_images"),
        clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
        phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
    ):
        self.db_path = db_path
        self.images_dir = images_dir
        self.clip_threshold = clip_threshold
        self.phash_threshold = phash_threshold

        self.graph = nx.DiGraph()
        self.embeddings: Dict[str, np.ndarray] = {}
        self.hashes: Dict[str, str] = {}

        self.device = "cpu"
        self.clip_model = None
        self.clip_processor = None
        self.clip_ready = False

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.load()

    def load_clip(self) -> None:
        if self.clip_ready:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        model.to(self.device)
        model.eval()
        self.clip_model = model
        self.clip_processor = processor
        self.clip_ready = True

    def get_clip_embedding(self, image_path: str) -> np.ndarray:
        if not self.clip_ready:
            self.load_clip()
        import torch

        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def get_phash(self, image_path: str) -> str:
        with Image.open(image_path) as image:
            return str(imagehash.phash(image.convert("RGB")))

    def add_user(self, username: str, bio: str = "", user_id: Optional[str] = None) -> str:
        username = username.strip()
        if not username:
            raise ValueError("Username is required")
        if any(
            d.get("type") == "user" and d.get("username", "").lower() == username.lower()
            for _, d in self.graph.nodes(data=True)
        ):
            raise ValueError(f"Username '{username}' already exists")

        user_id = user_id.strip() if user_id else f"U{uuid.uuid4().hex[:8]}"
        if user_id in self.graph:
            raise ValueError(f"User {user_id} already exists")

        self.graph.add_node(
            user_id,
            type="user",
            username=username,
            bio=bio.strip(),
            created_at=_now().isoformat(),
        )
        self.save()
        return user_id

    def add_post(
        self,
        user_id: str,
        image_path: str,
        caption: str = "",
        post_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[str, Optional[str]]:
        if user_id not in self.graph or self.graph.nodes[user_id].get("type") != "user":
            raise ValueError(f"User {user_id} does not exist")
        if not os.path.isfile(image_path):
            raise ValueError("Image file does not exist")

        post_id = post_id.strip() if post_id else f"P{uuid.uuid4().hex[:8]}"
        if post_id in self.graph:
            raise ValueError(f"Post {post_id} already exists")

        ext = os.path.splitext(image_path)[1] or ".png"
        stored_path = os.path.join(self.images_dir, f"{post_id}{ext}")
        shutil.copy2(image_path, stored_path)

        embedding = self.get_clip_embedding(stored_path)
        img_hash = self.get_phash(stored_path)
        ts = timestamp or _now()

        self.graph.add_node(
            post_id,
            type="post",
            image_path=stored_path,
            original_path=image_path,
            caption=caption.strip(),
            timestamp=ts.isoformat(),
            created_at=ts.isoformat(),
            likes=0,
        )
        self.graph.add_edge(user_id, post_id, type="posted", timestamp=ts.isoformat())
        self.embeddings[post_id] = embedding
        self.hashes[post_id] = img_hash

        source_id = self._link_repost(post_id, embedding, img_hash, ts)
        self.save()
        return post_id, source_id

    def follow(self, follower_id: str, followee_id: str) -> None:
        self._require_user(follower_id)
        self._require_user(followee_id)
        if follower_id == followee_id:
            raise ValueError("A user cannot follow themselves")
        if self.graph.has_edge(follower_id, followee_id) and self.graph.edges[follower_id, followee_id].get("type") == "follows":
            raise ValueError("Follow relationship already exists")
        self.graph.add_edge(follower_id, followee_id, type="follows")
        self.save()

    def like(self, user_id: str, post_id: str) -> int:
        self._require_user(user_id)
        self._require_post(post_id)
        if self.graph.has_edge(user_id, post_id) and self.graph.edges[user_id, post_id].get("type") == "likes":
            raise ValueError("User already liked this post")
        self.graph.add_edge(user_id, post_id, type="likes")
        likes = int(self.graph.nodes[post_id].get("likes", 0)) + 1
        self.graph.nodes[post_id]["likes"] = likes
        self.save()
        return likes

    def users(self) -> List[Dict]:
        rows = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "user":
                rows.append(
                    {
                        "id": node_id,
                        "username": data.get("username", node_id),
                        "bio": data.get("bio", ""),
                        "created_at": data.get("created_at", ""),
                        "followers": self._follower_count(node_id),
                        "posts": len(self.user_posts(node_id)),
                    }
                )
        rows.sort(key=lambda r: r["username"].lower())
        return rows

    def posts(self) -> List[Dict]:
        rows = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "post":
                user_id = self.post_author(node_id)
                username = self.graph.nodes[user_id]["username"] if user_id else "unknown"
                rows.append(
                    {
                        "id": node_id,
                        "user_id": user_id or "",
                        "username": username,
                        "caption": data.get("caption", ""),
                        "timestamp": data.get("timestamp", ""),
                        "likes": int(data.get("likes", 0)),
                        "image_path": data.get("image_path", ""),
                        "repost_of": self._direct_source(node_id),
                    }
                )
        rows.sort(key=lambda r: r["timestamp"])
        return rows

    def follows(self) -> List[Tuple[str, str, str, str]]:
        rows = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") == "follows":
                rows.append(
                    (
                        u,
                        self.graph.nodes[u].get("username", u),
                        v,
                        self.graph.nodes[v].get("username", v),
                    )
                )
        return rows

    def user_posts(self, user_id: str) -> List[str]:
        return [
            succ
            for succ in self.graph.successors(user_id)
            if self.graph.nodes[succ].get("type") == "post"
            and self.graph.edges[user_id, succ].get("type") == "posted"
        ]

    def post_author(self, post_id: str) -> Optional[str]:
        for pred in self.graph.predecessors(post_id):
            if self.graph.edges[pred, post_id].get("type") == "posted":
                return pred
        return None

    def find_similar(
        self,
        post_id: Optional[str] = None,
        image_path: Optional[str] = None,
        clip_threshold: Optional[float] = None,
        phash_threshold: Optional[int] = None,
    ) -> List[Match]:
        clip_cut = self.clip_threshold if clip_threshold is None else clip_threshold
        hash_cut = self.phash_threshold if phash_threshold is None else phash_threshold

        if post_id:
            self._require_post(post_id)
            query_emb = self.embeddings[post_id]
            query_hash = self.hashes[post_id]
            skip = {post_id}
        elif image_path:
            query_emb = self.get_clip_embedding(image_path)
            query_hash = self.get_phash(image_path)
            skip = set()
        else:
            raise ValueError("Provide a post_id or an image_path")

        matches: List[Match] = []
        for other_id, emb in self.embeddings.items():
            if other_id in skip:
                continue
            clip_sim = cosine_similarity(query_emb, emb)
            dist = phash_distance(query_hash, self.hashes[other_id])
            if clip_sim < clip_cut and dist > hash_cut:
                continue
            matches.append(self._match_from(other_id, clip_sim, dist))

        matches.sort(key=lambda m: (m.combined, m.clip_similarity, -m.phash_distance), reverse=True)
        return matches

    def trace_source(
        self,
        post_id: Optional[str] = None,
        image_path: Optional[str] = None,
        clip_threshold: Optional[float] = None,
        phash_threshold: Optional[int] = None,
    ) -> TraceResult:
        matches = self.find_similar(
            post_id=post_id,
            image_path=image_path,
            clip_threshold=clip_threshold,
            phash_threshold=phash_threshold,
        )

        cluster_ids = [m.post_id for m in matches]
        if post_id:
            cluster_ids = [post_id] + cluster_ids

        origin_id = None
        path: List[str] = []
        if post_id:
            origin_id, path = self._walk_to_origin(post_id)

        if origin_id is None and cluster_ids:
            origin_id = self._earliest_post(cluster_ids)
            path = [origin_id] if origin_id else []

        origin = None
        if origin_id:
            if post_id and origin_id == post_id and image_path is None:
                origin = self._match_from(origin_id, 1.0, 0)
            else:
                origin = next((m for m in matches if m.post_id == origin_id), None)
                if origin is None:
                    if image_path:
                        clip_sim = cosine_similarity(
                            self.get_clip_embedding(image_path),
                            self.embeddings[origin_id],
                        )
                        dist = phash_distance(self.get_phash(image_path), self.hashes[origin_id])
                    elif post_id:
                        clip_sim = cosine_similarity(self.embeddings[post_id], self.embeddings[origin_id])
                        dist = phash_distance(self.hashes[post_id], self.hashes[origin_id])
                    else:
                        clip_sim, dist = 1.0, 0
                    origin = self._match_from(origin_id, clip_sim, dist)
            origin.is_origin = True

        centrality_user = self.find_source_by_centrality(cluster_ids) if len(cluster_ids) >= 2 else None
        centrality_name = None
        if centrality_user:
            centrality_name = self.graph.nodes[centrality_user].get("username", centrality_user)

        reasoning = self._reason(origin, matches, path, centrality_name)
        return TraceResult(
            query_post_id=post_id,
            origin=origin,
            matches=matches,
            path=path,
            reasoning=reasoning,
            centrality_user_id=centrality_user,
            centrality_username=centrality_name,
        )

    def find_source_by_centrality(self, post_ids: List[str]) -> Optional[str]:
        users = set()
        for pid in post_ids:
            author = self.post_author(pid)
            if author:
                users.add(author)
        if not users:
            return None
        if len(users) == 1:
            return next(iter(users))

        subgraph = self.graph.subgraph(users).copy()
        follow_edges = [
            (u, v)
            for u, v, d in subgraph.edges(data=True)
            if d.get("type") == "follows"
        ]
        if not follow_edges:
            return self.post_author(self._earliest_post(post_ids) or post_ids[0])

        follow_graph = nx.DiGraph()
        follow_graph.add_nodes_from(users)
        follow_graph.add_edges_from(follow_edges)
        ranks = nx.pagerank(follow_graph, alpha=0.85)
        return max(ranks, key=ranks.get)

    def save(self) -> None:
        payload = {
            "graph": self.graph,
            "embeddings": self.embeddings,
            "hashes": self.hashes,
        }
        tmp_path = self.db_path + ".tmp"
        with open(tmp_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, self.db_path)

    def load(self) -> None:
        if not os.path.isfile(self.db_path):
            return
        try:
            with open(self.db_path, "rb") as handle:
                payload = pickle.load(handle)
            self.graph = payload.get("graph", nx.DiGraph())
            self.embeddings = payload.get("embeddings", {})
            self.hashes = payload.get("hashes", {})
        except Exception as exc:
            print(f"Could not load database ({exc}). Starting empty.")
            self.graph = nx.DiGraph()
            self.embeddings = {}
            self.hashes = {}

    def _require_user(self, user_id: str) -> None:
        if user_id not in self.graph or self.graph.nodes[user_id].get("type") != "user":
            raise ValueError(f"User {user_id} does not exist")

    def _require_post(self, post_id: str) -> None:
        if post_id not in self.graph or self.graph.nodes[post_id].get("type") != "post":
            raise ValueError(f"Post {post_id} does not exist")

    def _follower_count(self, user_id: str) -> int:
        return sum(
            1
            for pred in self.graph.predecessors(user_id)
            if self.graph.edges[pred, user_id].get("type") == "follows"
        )

    def _direct_source(self, post_id: str) -> Optional[str]:
        for succ in self.graph.successors(post_id):
            if self.graph.edges[post_id, succ].get("type") == "reposted_from":
                return succ
        return None

    def _link_repost(
        self,
        post_id: str,
        embedding: np.ndarray,
        img_hash: str,
        timestamp: datetime,
    ) -> Optional[str]:
        best_id = None
        best_ts = None
        for other_id, other_emb in self.embeddings.items():
            if other_id == post_id:
                continue
            clip_sim = cosine_similarity(embedding, other_emb)
            dist = phash_distance(img_hash, self.hashes[other_id])
            if clip_sim < self.clip_threshold and dist > self.phash_threshold:
                continue
            other_ts = _parse_ts(self.graph.nodes[other_id].get("timestamp"))
            if other_ts >= timestamp:
                continue
            if best_ts is None or other_ts < best_ts:
                best_id = other_id
                best_ts = other_ts
        if best_id:
            self.graph.add_edge(post_id, best_id, type="reposted_from")
        return best_id

    def _walk_to_origin(self, post_id: str) -> Tuple[str, List[str]]:
        current = post_id
        path = [current]
        visited = {current}
        while True:
            source = self._direct_source(current)
            if not source or source in visited:
                break
            path.append(source)
            visited.add(source)
            current = source
        path.reverse()
        return current, path

    def _earliest_post(self, post_ids: List[str]) -> Optional[str]:
        dated = []
        for pid in post_ids:
            if pid in self.graph and self.graph.nodes[pid].get("type") == "post":
                dated.append(( _parse_ts(self.graph.nodes[pid].get("timestamp")), pid))
        if not dated:
            return None
        dated.sort()
        return dated[0][1]

    def _match_from(self, post_id: str, clip_sim: float, phash_dist: int) -> Match:
        data = self.graph.nodes[post_id]
        user_id = self.post_author(post_id) or ""
        username = self.graph.nodes[user_id].get("username", user_id) if user_id else "unknown"
        return Match(
            post_id=post_id,
            user_id=user_id,
            username=username,
            clip_similarity=clip_sim,
            phash_distance=phash_dist,
            combined=combined_score(clip_sim, phash_dist),
            timestamp=_parse_ts(data.get("timestamp")),
            image_path=data.get("image_path", ""),
            caption=data.get("caption", ""),
        )

    def _reason(
        self,
        origin: Optional[Match],
        matches: List[Match],
        path: List[str],
        centrality_name: Optional[str],
    ) -> str:
        if origin is None:
            return "No visually similar posts were found, so this image looks original in the current network."
        hops = max(0, len(path) - 1)
        parts = [
            f"Likely original is post {origin.post_id} by {origin.username}.",
            f"CLIP cosine {origin.clip_similarity:.3f}, pHash distance {origin.phash_distance}, combined {origin.combined:.3f}.",
            f"Posted at {origin.timestamp.isoformat(sep=' ', timespec='seconds')}.",
        ]
        if hops:
            parts.append(f"Repost chain has {hops} hop(s): {' -> '.join(path)}.")
        else:
            parts.append("No earlier near-duplicate sits behind this post.")
        if matches:
            parts.append(f"{len(matches)} near-match(es) ranked by similarity.")
        if centrality_name:
            parts.append(f"PageRank on the follow subgraph of matching posters points at {centrality_name}.")
        return " ".join(parts)
