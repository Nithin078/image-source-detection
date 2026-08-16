"""
Image source detection: CLIP embeddings + perceptual hash + NetworkX graph.

The social network is a directed multigraph of users and posts. When a new
image is posted, it is compared against stored fingerprints. Near-matches
that pass a strict link test get a reposted_from edge. Source tracing ranks
candidates by time and similarity; PageRank is only an amplifier hint.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import imagehash
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    from PIL.ExifTags import IFD

    EXIF_IFD = IFD.Exif
except Exception:
    EXIF_IFD = 0x8769

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_CLIP_THRESHOLD = 0.85
DEFAULT_LINK_CLIP_THRESHOLD = 0.90
DEFAULT_PHASH_THRESHOLD = 5
DEFAULT_LINK_COMBINED_THRESHOLD = 0.88
PHASH_BITS = 64
_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(_ROOT, "data")


def _now() -> datetime:
    return datetime.now()


def _parse_ts(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return _now()


def parse_exif_datetime(raw: str) -> Optional[datetime]:
    text = str(raw).strip().split(".")[0]
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def image_timestamp(image_path: str) -> Tuple[datetime, str]:
    """Prefer EXIF DateTimeOriginal, then file mtime, then now."""
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            if exif:
                values = []
                for tag in (36867, 36868, 306):
                    value = exif.get(tag)
                    if value:
                        values.append(value)
                try:
                    ifd = exif.get_ifd(EXIF_IFD)
                    for tag in (36867, 36868, 306):
                        value = ifd.get(tag)
                        if value:
                            values.append(value)
                except Exception:
                    pass
                for raw in values:
                    parsed = parse_exif_datetime(str(raw))
                    if parsed:
                        return parsed, "exif"
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(image_path)), "mtime"
    except OSError:
        return _now(), "now"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def phash_distance(hash_a: str, hash_b: str) -> int:
    return int(imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b))


def hash_similarity(distance: int) -> float:
    return max(0.0, 1.0 - (int(distance) / PHASH_BITS))


def combined_score(clip_sim: float, phash_dist: int) -> float:
    return 0.7 * clip_sim + 0.3 * hash_similarity(phash_dist)


def is_search_match(
    clip_sim: float,
    phash_dist: int,
    clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
    phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
) -> bool:
    return clip_sim >= clip_threshold or int(phash_dist) <= phash_threshold


def is_link_match(
    clip_sim: float,
    phash_dist: int,
    link_clip_threshold: float = DEFAULT_LINK_CLIP_THRESHOLD,
    phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
    link_combined_threshold: float = DEFAULT_LINK_COMBINED_THRESHOLD,
) -> bool:
    dist = int(phash_dist)
    if clip_sim >= link_clip_threshold and dist <= phash_threshold:
        return True
    return combined_score(clip_sim, dist) >= link_combined_threshold


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
    used_edge_fallback: bool = False


class ImageSourceDetector:
    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        db_path: Optional[str] = None,
        images_dir: Optional[str] = None,
        clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
        phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
        link_clip_threshold: float = DEFAULT_LINK_CLIP_THRESHOLD,
        link_combined_threshold: float = DEFAULT_LINK_COMBINED_THRESHOLD,
    ):
        if db_path:
            data_dir = os.path.dirname(os.path.abspath(db_path)) or data_dir
        self.data_dir = data_dir
        self.graph_path = os.path.join(data_dir, "graph.json")
        self.hashes_path = os.path.join(data_dir, "hashes.json")
        self.emb_dir = os.path.join(data_dir, "embeddings")
        self.images_dir = images_dir or os.path.join(data_dir, "stored_images")
        self.sample_dir = os.path.join(data_dir, "sample_images")
        self.legacy_pkl = os.path.join(data_dir, "image_network.pkl")

        self.clip_threshold = clip_threshold
        self.phash_threshold = phash_threshold
        self.link_clip_threshold = link_clip_threshold
        self.link_combined_threshold = link_combined_threshold

        self.graph = nx.MultiDiGraph()
        self.embeddings: Dict[str, np.ndarray] = {}
        self.hashes: Dict[str, str] = {}

        self.device = "cpu"
        self.clip_model = None
        self.clip_processor = None
        self.clip_ready = False
        self._lock = threading.RLock()

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.emb_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        self.load()

    def load_clip(self) -> None:
        with self._lock:
            if self.clip_ready:
                return
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
            try:
                processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, use_fast=True)
            except TypeError:
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

        with self._lock:
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
        with self._lock:
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
        if not os.path.isfile(image_path):
            raise ValueError("Image file does not exist")

        embedding = self.get_clip_embedding(image_path)
        img_hash = self.get_phash(image_path)
        if timestamp is None:
            ts, ts_source = image_timestamp(image_path)
        else:
            ts, ts_source = timestamp, "manual"

        with self._lock:
            if user_id not in self.graph or self.graph.nodes[user_id].get("type") != "user":
                raise ValueError(f"User {user_id} does not exist")

            post_id = post_id.strip() if post_id else f"P{uuid.uuid4().hex[:8]}"
            if post_id in self.graph:
                raise ValueError(f"Post {post_id} already exists")

            ext = os.path.splitext(image_path)[1] or ".png"
            stored_path = os.path.join(self.images_dir, f"{post_id}{ext}")
            if os.path.abspath(image_path) != os.path.abspath(stored_path):
                shutil.copy2(image_path, stored_path)

            self.graph.add_node(
                post_id,
                type="post",
                image_path=stored_path,
                original_path=image_path,
                caption=caption.strip(),
                timestamp=ts.isoformat(),
                timestamp_source=ts_source,
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
        with self._lock:
            self._require_user(follower_id)
            self._require_user(followee_id)
            if follower_id == followee_id:
                raise ValueError("A user cannot follow themselves")
            if self._has_typed_edge(follower_id, followee_id, "follows"):
                raise ValueError("Follow relationship already exists")
            self.graph.add_edge(follower_id, followee_id, type="follows")
            self.save()

    def like(self, user_id: str, post_id: str) -> int:
        with self._lock:
            self._require_user(user_id)
            self._require_post(post_id)
            if self._has_typed_edge(user_id, post_id, "likes"):
                raise ValueError("User already liked this post")
            self.graph.add_edge(user_id, post_id, type="likes")
            likes = int(self.graph.nodes[post_id].get("likes", 0)) + 1
            self.graph.nodes[post_id]["likes"] = likes
            self.save()
            return likes

    def unfollow(self, follower_id: str, followee_id: str) -> None:
        with self._lock:
            if not self._remove_typed_edges(follower_id, followee_id, "follows"):
                raise ValueError("That follow relationship does not exist")
            self.save()

    def unlike(self, user_id: str, post_id: str) -> int:
        with self._lock:
            self._require_post(post_id)
            if not self._remove_typed_edges(user_id, post_id, "likes"):
                raise ValueError("That like does not exist")
            likes = max(0, int(self.graph.nodes[post_id].get("likes", 0)) - 1)
            self.graph.nodes[post_id]["likes"] = likes
            self.save()
            return likes

    def delete_post(self, post_id: str) -> None:
        with self._lock:
            self._require_post(post_id)
            image_path = self.graph.nodes[post_id].get("image_path")
            self.graph.remove_node(post_id)
            self.embeddings.pop(post_id, None)
            self.hashes.pop(post_id, None)
            if image_path and os.path.isfile(image_path):
                try:
                    os.remove(image_path)
                except OSError:
                    pass
            self.save()

    def delete_user(self, user_id: str) -> int:
        with self._lock:
            self._require_user(user_id)
            post_ids = list(self.user_posts(user_id))
        for post_id in post_ids:
            self.delete_post(post_id)
        with self._lock:
            if user_id in self.graph:
                self.graph.remove_node(user_id)
                self.save()
        return len(post_ids)

    def reset(self) -> None:
        with self._lock:
            self.graph = nx.MultiDiGraph()
            self.embeddings = {}
            self.hashes = {}
            for folder in (self.images_dir, self.emb_dir):
                if not os.path.isdir(folder):
                    continue
                for name in os.listdir(folder):
                    if name == ".gitkeep":
                        continue
                    path = os.path.join(folder, name)
                    if os.path.isfile(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass
            self.save()

    def users(self) -> List[Dict]:
        with self._lock:
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
        with self._lock:
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
                            "timestamp_source": data.get("timestamp_source", ""),
                            "likes": int(data.get("likes", 0)),
                            "image_path": data.get("image_path", ""),
                            "repost_of": self._direct_source(node_id),
                        }
                    )
            rows.sort(key=lambda r: r["timestamp"])
            return rows

    def follows(self) -> List[Tuple[str, str, str, str]]:
        with self._lock:
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
        return list(self._out_neighbors(user_id, "posted"))

    def post_author(self, post_id: str) -> Optional[str]:
        return next(self._in_neighbors(post_id, "posted"), None)

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
            with self._lock:
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

        with self._lock:
            matches: List[Match] = []
            for other_id, emb in self.embeddings.items():
                if other_id in skip:
                    continue
                clip_sim = cosine_similarity(query_emb, emb)
                dist = phash_distance(query_hash, self.hashes[other_id])
                if not is_search_match(clip_sim, dist, clip_cut, hash_cut):
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

        with self._lock:
            query_match = None
            if post_id:
                query_match = self._match_from(post_id, 1.0, 0)

            linkable = [m for m in matches if self._is_link_match(m.clip_similarity, m.phash_distance)]
            pool = linkable if linkable else list(matches)
            if query_match:
                pool = [query_match] + [m for m in pool if m.post_id != post_id]

            if not pool:
                return TraceResult(
                    query_post_id=post_id,
                    origin=None,
                    matches=matches,
                    reasoning="No visually similar posts were found, so this image looks original in the current network.",
                )

            def sort_key(match: Match):
                length = 0
                if post_id:
                    path = self._repost_path(match.post_id, post_id)
                    length = (len(path) - 1) if path else 10_000
                return (match.timestamp, length, -match.combined)

            pool.sort(key=sort_key)
            chosen = pool[0]

            used_fallback = False
            if post_id:
                walked_id, walked_path = self._walk_to_origin(post_id)
                if walked_id == chosen.post_id and len(walked_path) > 1:
                    path = walked_path
                else:
                    reconstructed = self._repost_path(chosen.post_id, post_id)
                    if reconstructed:
                        path = reconstructed
                    elif chosen.post_id == post_id:
                        path = [post_id]
                    else:
                        path = [chosen.post_id, post_id]
                        used_fallback = walked_id == post_id and chosen.post_id != post_id
            else:
                path = [chosen.post_id]

            if query_match and chosen.post_id == post_id:
                origin = query_match
            else:
                origin = next((m for m in matches if m.post_id == chosen.post_id), chosen)
            origin.is_origin = True

            cluster_ids = [m.post_id for m in matches]
            if post_id:
                cluster_ids = [post_id] + cluster_ids
            centrality_user = self.find_source_by_centrality(cluster_ids) if len(set(cluster_ids)) >= 2 else None
            centrality_name = None
            if centrality_user:
                centrality_name = self.graph.nodes[centrality_user].get("username", centrality_user)

            reasoning = self._reason(origin, matches, path, centrality_name, used_fallback)
            return TraceResult(
                query_post_id=post_id,
                origin=origin,
                matches=matches,
                path=path,
                reasoning=reasoning,
                centrality_user_id=centrality_user,
                centrality_username=centrality_name,
                used_edge_fallback=used_fallback,
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

        follow_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if d.get("type") == "follows" and u in users and v in users
        ]
        if not follow_edges:
            earliest = self._earliest_post(post_ids)
            return self.post_author(earliest) if earliest else None

        follow_graph = nx.DiGraph()
        follow_graph.add_nodes_from(users)
        follow_graph.add_edges_from(follow_edges)
        ranks = nx.pagerank(follow_graph, alpha=0.85)
        return max(ranks, key=ranks.get)

    def create_sample_images(self) -> Dict[str, str]:
        os.makedirs(self.sample_dir, exist_ok=True)
        original_path = os.path.join(self.sample_dir, "original.png")
        near_path = os.path.join(self.sample_dir, "near_copy.png")
        other_path = os.path.join(self.sample_dir, "unrelated.png")

        original = Image.new("RGB", (256, 256), color=(30, 90, 200))
        draw = ImageDraw.Draw(original)
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except OSError:
            font = ImageFont.load_default()
        draw.rectangle((40, 40, 216, 216), outline=(255, 255, 255), width=6)
        draw.text((58, 110), "ORIGINAL", fill=(255, 255, 255), font=font)
        original.save(original_path)

        near = original.crop((20, 20, 236, 236)).resize((256, 256)).filter(ImageFilter.GaussianBlur(1.2))
        near.save(near_path)

        other = Image.new("RGB", (256, 256), color=(190, 40, 40))
        other_draw = ImageDraw.Draw(other)
        other_draw.ellipse((48, 48, 208, 208), fill=(255, 200, 60))
        other_draw.text((78, 112), "OTHER", fill=(40, 20, 20), font=font)
        other.save(other_path)

        return {"original": original_path, "near": near_path, "other": other_path}

    def load_sample_network(self) -> Dict[str, object]:
        if not self.clip_ready:
            self.load_clip()

        images = self.create_sample_images()
        created_users = {}

        def ensure_user(name: str, bio: str) -> str:
            for user in self.users():
                if user["username"].lower() == name:
                    return user["id"]
            return self.add_user(name, bio)

        bob = ensure_user("bob", "original poster")
        alice = ensure_user("alice", "reposter")
        carol = ensure_user("carol", "unrelated poster")
        created_users = {"bob": bob, "alice": alice, "carol": carol}

        try:
            self.follow(alice, bob)
        except ValueError:
            pass

        now = _now()
        original_id, _ = self.add_post(
            bob,
            images["original"],
            caption="[sample] original photo",
            timestamp=now - timedelta(hours=2),
        )
        near_id, linked = self.add_post(
            alice,
            images["near"],
            caption="[sample] screenshot / crop",
            timestamp=now - timedelta(hours=1),
        )
        other_id, _ = self.add_post(
            carol,
            images["other"],
            caption="[sample] unrelated image",
            timestamp=now,
        )
        return {
            "users": created_users,
            "posts": {"original": original_id, "near": near_id, "other": other_id},
            "linked": linked,
        }

    def save(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.emb_dir, exist_ok=True)
        payload = {
            "directed": True,
            "multigraph": True,
            "nodes": [],
            "edges": [],
        }
        for node_id, data in self.graph.nodes(data=True):
            row = {"id": node_id}
            for key, value in data.items():
                row[key] = value
            payload["nodes"].append(row)
        for source, target, data in self.graph.edges(data=True):
            row = {"source": source, "target": target}
            for key, value in data.items():
                row[key] = value
            payload["edges"].append(row)

        tmp_graph = self.graph_path + ".tmp"
        with open(tmp_graph, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_graph, self.graph_path)

        tmp_hashes = self.hashes_path + ".tmp"
        with open(tmp_hashes, "w", encoding="utf-8") as handle:
            json.dump(self.hashes, handle, indent=2)
        os.replace(tmp_hashes, self.hashes_path)

        live = set(self.embeddings)
        for post_id, embedding in self.embeddings.items():
            np.save(os.path.join(self.emb_dir, f"{post_id}.npy"), embedding)
        for name in os.listdir(self.emb_dir):
            if name.endswith(".npy") and name[:-4] not in live:
                os.remove(os.path.join(self.emb_dir, name))

    def load(self) -> None:
        if os.path.isfile(self.graph_path):
            self._load_json()
            return
        if os.path.isfile(self.legacy_pkl):
            self._load_legacy_pickle()
            try:
                self.save()
            except Exception as exc:
                print(f"Migrated pickle but could not rewrite JSON store ({exc}).")

    def _load_json(self) -> None:
        try:
            with open(self.graph_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(f"Could not load graph.json ({exc}). Starting empty.")
            return

        graph = nx.MultiDiGraph()
        for row in payload.get("nodes", []):
            node_id = row.get("id")
            if not node_id:
                continue
            attrs = {k: v for k, v in row.items() if k != "id"}
            graph.add_node(node_id, **attrs)
        for row in payload.get("edges", []):
            source = row.get("source")
            target = row.get("target")
            if source is None or target is None:
                continue
            attrs = {k: v for k, v in row.items() if k not in ("source", "target")}
            graph.add_edge(source, target, **attrs)
        self.graph = graph

        self.hashes = {}
        if os.path.isfile(self.hashes_path):
            try:
                with open(self.hashes_path, "r", encoding="utf-8") as handle:
                    self.hashes = json.load(handle)
            except Exception as exc:
                print(f"Could not load hashes.json ({exc}).")

        self.embeddings = {}
        if os.path.isdir(self.emb_dir):
            for name in os.listdir(self.emb_dir):
                if not name.endswith(".npy"):
                    continue
                post_id = name[:-4]
                if post_id not in self.graph:
                    continue
                try:
                    self.embeddings[post_id] = np.load(os.path.join(self.emb_dir, name))
                except Exception as exc:
                    print(f"Could not load embedding {name} ({exc}).")

    def _load_legacy_pickle(self) -> None:
        try:
            with open(self.legacy_pkl, "rb") as handle:
                payload = pickle.load(handle)
            graph = payload.get("graph", nx.DiGraph())
            if not isinstance(graph, nx.MultiDiGraph):
                multi = nx.MultiDiGraph()
                multi.add_nodes_from(graph.nodes(data=True))
                for u, v, data in graph.edges(data=True):
                    multi.add_edge(u, v, **data)
                graph = multi
            self.graph = graph
            self.embeddings = payload.get("embeddings", {})
            self.hashes = payload.get("hashes", {})
        except Exception as exc:
            print(f"Could not load legacy pickle ({exc}). Starting empty.")
            self.graph = nx.MultiDiGraph()
            self.embeddings = {}
            self.hashes = {}

    def _require_user(self, user_id: str) -> None:
        if user_id not in self.graph or self.graph.nodes[user_id].get("type") != "user":
            raise ValueError(f"User {user_id} does not exist")

    def _require_post(self, post_id: str) -> None:
        if post_id not in self.graph or self.graph.nodes[post_id].get("type") != "post":
            raise ValueError(f"Post {post_id} does not exist")

    def _typed_edges_between(self, source: str, target: str) -> List[Dict]:
        data = self.graph.get_edge_data(source, target)
        if not data:
            return []
        return list(data.values())

    def _has_typed_edge(self, source: str, target: str, etype: str) -> bool:
        return any(edge.get("type") == etype for edge in self._typed_edges_between(source, target))

    def _remove_typed_edges(self, source: str, target: str, etype: str) -> int:
        data = self.graph.get_edge_data(source, target)
        if not data:
            return 0
        removed = 0
        for key in list(data.keys()):
            if data[key].get("type") == etype:
                self.graph.remove_edge(source, target, key)
                removed += 1
        return removed

    def _out_neighbors(self, source: str, etype: str) -> Iterable[str]:
        for _, target, data in self.graph.out_edges(source, data=True):
            if data.get("type") == etype:
                yield target

    def _in_neighbors(self, target: str, etype: str) -> Iterable[str]:
        for source, _, data in self.graph.in_edges(target, data=True):
            if data.get("type") == etype:
                yield source

    def _follower_count(self, user_id: str) -> int:
        return sum(1 for _ in self._in_neighbors(user_id, "follows"))

    def _direct_source(self, post_id: str) -> Optional[str]:
        return next(self._out_neighbors(post_id, "reposted_from"), None)

    def _is_link_match(self, clip_sim: float, phash_dist: int) -> bool:
        return is_link_match(
            clip_sim,
            phash_dist,
            self.link_clip_threshold,
            self.phash_threshold,
            self.link_combined_threshold,
        )

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
            if not self._is_link_match(clip_sim, dist):
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

    def _repost_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for newer, older, data in self.graph.edges(data=True):
            if data.get("type") == "reposted_from":
                graph.add_edge(older, newer)
        return graph

    def _repost_path(self, origin_id: str, query_id: str) -> Optional[List[str]]:
        if origin_id == query_id:
            return [origin_id]
        try:
            return nx.shortest_path(self._repost_graph(), origin_id, query_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _earliest_post(self, post_ids: List[str]) -> Optional[str]:
        dated = []
        for pid in post_ids:
            if pid in self.graph and self.graph.nodes[pid].get("type") == "post":
                dated.append((_parse_ts(self.graph.nodes[pid].get("timestamp")), pid))
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
            clip_similarity=float(clip_sim),
            phash_distance=int(phash_dist),
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
        used_fallback: bool,
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
        if used_fallback:
            parts.append("No stored repost edge pointed here, so the earliest strong near-match was used.")
        if matches:
            parts.append(f"{len(matches)} near-match(es) ranked by similarity.")
        if centrality_name:
            parts.append(f"Likely amplifier (follow-graph PageRank): {centrality_name}.")
        return " ".join(parts)
