"""
Optional DGL heterogeneous-graph backend.

The GUI uses NetworkX. This module keeps the internship DGL experiment as a
small, runnable script: users, posts, follows, and CLIP-based source lookup
on a DGL heterograph.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from detector import cosine_similarity


def _load_dgl():
    try:
        import dgl
    except Exception as exc:
        raise ImportError(
            "DGL is optional and could not be imported. Install a matching "
            "`pip install dgl` build if you want this backend. The main GUI "
            "does not need it."
        ) from exc
    return dgl


class DGLImageGraph:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        dgl = _load_dgl()
        self.user_ids: List[str] = []
        self.usernames: List[str] = []
        self.post_ids: List[str] = []
        self.image_paths: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.graph = dgl.heterograph(
            {
                ("user", "follows", "user"): ([], []),
                ("user", "posted", "post"): ([], []),
                ("post", "posted_by", "user"): ([], []),
            }
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        try:
            self.processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
        except TypeError:
            self.processor = CLIPProcessor.from_pretrained(model_id)
        self.device = device
        self.model.eval()

    def embed(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def add_user(self, user_id: str, username: str) -> int:
        self.user_ids.append(user_id)
        self.usernames.append(username)
        self.graph.add_nodes(1, ntype="user")
        return len(self.user_ids) - 1

    def add_post(self, user_idx: int, post_id: str, image_path: str) -> int:
        embedding = self.embed(image_path)
        self.post_ids.append(post_id)
        self.image_paths.append(image_path)
        self.embeddings.append(embedding)
        self.graph.add_nodes(1, ntype="post")
        post_idx = self.graph.num_nodes("post") - 1
        self.graph.add_edges(user_idx, post_idx, etype="posted")
        self.graph.add_edges(post_idx, user_idx, etype="posted_by")
        return post_idx

    def follow(self, follower_idx: int, followee_idx: int) -> None:
        self.graph.add_edges(follower_idx, followee_idx, etype="follows")

    def trace(self, image_path: str, threshold: float = 0.85) -> Optional[Dict]:
        query = self.embed(image_path)
        best_idx = -1
        best_sim = -1.0
        for idx, emb in enumerate(self.embeddings):
            sim = cosine_similarity(query, emb)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx < 0 or best_sim < threshold:
            return None
        authors = self.graph.successors(best_idx, etype="posted_by")
        if len(authors) == 0:
            return None
        user_idx = int(authors[0])
        return {
            "post_id": self.post_ids[best_idx],
            "image_path": self.image_paths[best_idx],
            "similarity": best_sim,
            "username": self.usernames[user_idx],
            "user_id": self.user_ids[user_idx],
        }


def demo(query_image: str, stored_image: str) -> None:
    graph = DGLImageGraph()
    alice = graph.add_user("U_alice", "alice")
    bob = graph.add_user("U_bob", "bob")
    graph.follow(alice, bob)
    graph.add_post(bob, "P_original", stored_image)
    result = graph.trace(query_image)
    if result:
        print("Original source found")
        print(f"  posted by : {result['username']}")
        print(f"  post id   : {result['post_id']}")
        print(f"  CLIP sim  : {result['similarity']:.4f}")
    else:
        print("No source found above threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGL + CLIP source lookup demo")
    parser.add_argument("stored_image", help="Image stored as the original post")
    parser.add_argument("query_image", help="Query image to trace")
    args = parser.parse_args()
    demo(args.query_image, args.stored_image)
