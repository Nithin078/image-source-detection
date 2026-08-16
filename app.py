"""Tkinter GUI for the image source detection prototype."""

from __future__ import annotations

import os
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from detector import ImageSourceDetector

APP_TITLE = "Image Source Detection"
THUMB_SIZE = (220, 220)


class ImageSourceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1280x860")
        self.root.minsize(980, 700)

        self.detector = ImageSourceDetector()
        self.status = tk.StringVar(value="Loading CLIP (ViT-B/32)...")
        self.photo_cache = []
        self.graph_canvas: Optional[FigureCanvasTkAgg] = None
        self._working = False
        self.last_query_path = None
        self.last_query_post = None

        self._build()
        self._set_busy(True)
        threading.Thread(target=self._load_model, daemon=True).start()

    def _build(self) -> None:
        header = ttk.Frame(self.root, padding=(12, 10, 12, 4))
        header.pack(fill="x")
        titles = ttk.Frame(header)
        titles.pack(side="left", fill="x", expand=True)
        ttk.Label(titles, text=APP_TITLE, font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(
            titles,
            text="CLIP embeddings + pHash near-duplicates + NetworkX social graph",
            font=("Segoe UI", 10),
        ).pack(anchor="w")
        ttk.Button(header, text="Load sample network", command=self._load_sample).pack(side="right", padx=(8, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=6)

        self._users_tab()
        self._posts_tab()
        self._relationships_tab()
        self._analysis_tab()
        self._graph_tab()

        status_bar = ttk.Frame(self.root, padding=(10, 4, 10, 8))
        status_bar.pack(fill="x")
        ttk.Label(status_bar, textvariable=self.status).pack(side="left")
        ttk.Button(status_bar, text="Save database", command=self._save).pack(side="right")

    def _users_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Users")

        form = ttk.LabelFrame(frame, text="Add user", padding=10)
        form.pack(fill="x")
        ttk.Label(form, text="Username").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.username_entry = ttk.Entry(form, width=24)
        self.username_entry.grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(form, text="Bio").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        self.bio_entry = ttk.Entry(form, width=40)
        self.bio_entry.grid(row=0, column=3, padx=4, pady=4)
        ttk.Button(form, text="Add user", command=self._add_user).grid(row=0, column=4, padx=8)

        table = ttk.Frame(frame)
        table.pack(fill="both", expand=True, pady=(10, 0))
        cols = ("id", "username", "bio", "followers", "posts")
        self.users_tree = ttk.Treeview(table, columns=cols, show="headings", height=16)
        headings = {
            "id": "User ID",
            "username": "Username",
            "bio": "Bio",
            "followers": "Followers",
            "posts": "Posts",
        }
        widths = {"id": 140, "username": 160, "bio": 360, "followers": 90, "posts": 80}
        for col in cols:
            self.users_tree.heading(col, text=headings[col])
            self.users_tree.column(col, width=widths[col], anchor="w")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.users_tree.yview)
        self.users_tree.configure(yscrollcommand=scroll.set)
        self.users_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _posts_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Posts")

        form = ttk.LabelFrame(frame, text="Add post", padding=10)
        form.pack(fill="x")
        ttk.Label(form, text="Poster").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.post_user_combo = ttk.Combobox(form, width=28, state="readonly")
        self.post_user_combo.grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(form, text="Image").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        self.image_path_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.image_path_var, width=42).grid(row=0, column=3, padx=4, pady=4)
        ttk.Button(form, text="Browse", command=self._browse_image).grid(row=0, column=4, padx=4)
        ttk.Label(form, text="Caption").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.caption_entry = ttk.Entry(form, width=70)
        self.caption_entry.grid(row=1, column=1, columnspan=3, sticky="ew", padx=4, pady=4)
        ttk.Label(form, text="Timestamp").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        self.timestamp_entry = ttk.Entry(form, width=28)
        self.timestamp_entry.grid(row=2, column=1, padx=4, pady=4, sticky="w")
        ttk.Label(form, text="Optional: YYYY-MM-DD HH:MM:SS (blank = now)").grid(
            row=2, column=2, columnspan=2, sticky="w", padx=4, pady=4
        )
        ttk.Button(form, text="Add post", command=self._add_post).grid(row=1, column=4, padx=4)

        table = ttk.Frame(frame)
        table.pack(fill="both", expand=True, pady=(10, 0))
        cols = ("id", "user", "caption", "timestamp", "likes", "repost_of")
        self.posts_tree = ttk.Treeview(table, columns=cols, show="headings", height=16)
        headings = {
            "id": "Post ID",
            "user": "Posted by",
            "caption": "Caption",
            "timestamp": "Timestamp",
            "likes": "Likes",
            "repost_of": "Repost of",
        }
        widths = {"id": 130, "user": 160, "caption": 280, "timestamp": 180, "likes": 70, "repost_of": 130}
        for col in cols:
            self.posts_tree.heading(col, text=headings[col])
            self.posts_tree.column(col, width=widths[col], anchor="w")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.posts_tree.yview)
        self.posts_tree.configure(yscrollcommand=scroll.set)
        self.posts_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.posts_tree.bind("<Double-1>", self._view_post)

    def _relationships_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Relationships")

        follow = ttk.LabelFrame(frame, text="Follow", padding=10)
        follow.pack(fill="x")
        ttk.Label(follow, text="Follower").grid(row=0, column=0, padx=4, pady=4)
        self.follower_combo = ttk.Combobox(follow, width=28, state="readonly")
        self.follower_combo.grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(follow, text="Follows").grid(row=0, column=2, padx=4, pady=4)
        self.followee_combo = ttk.Combobox(follow, width=28, state="readonly")
        self.followee_combo.grid(row=0, column=3, padx=4, pady=4)
        ttk.Button(follow, text="Add follow", command=self._add_follow).grid(row=0, column=4, padx=8)

        like = ttk.LabelFrame(frame, text="Like a post", padding=10)
        like.pack(fill="x", pady=(8, 0))
        ttk.Label(like, text="User").grid(row=0, column=0, padx=4, pady=4)
        self.like_user_combo = ttk.Combobox(like, width=28, state="readonly")
        self.like_user_combo.grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(like, text="Post").grid(row=0, column=2, padx=4, pady=4)
        self.like_post_combo = ttk.Combobox(like, width=28, state="readonly")
        self.like_post_combo.grid(row=0, column=3, padx=4, pady=4)
        ttk.Button(like, text="Like", command=self._add_like).grid(row=0, column=4, padx=8)

        table = ttk.LabelFrame(frame, text="Follow edges", padding=10)
        table.pack(fill="both", expand=True, pady=(10, 0))
        self.rel_tree = ttk.Treeview(table, columns=("follower", "followee"), show="headings")
        self.rel_tree.heading("follower", text="Follower")
        self.rel_tree.heading("followee", text="Follows")
        self.rel_tree.column("follower", width=360)
        self.rel_tree.column("followee", width=360)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.rel_tree.yview)
        self.rel_tree.configure(yscrollcommand=scroll.set)
        self.rel_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _analysis_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Trace source")

        controls = ttk.LabelFrame(frame, text="Query", padding=10)
        controls.pack(fill="x")
        ttk.Label(controls, text="Post").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.analysis_post_combo = ttk.Combobox(controls, width=28, state="readonly")
        self.analysis_post_combo.grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(controls, text="Trace selected post", command=self._trace_post).grid(row=0, column=2, padx=6)
        ttk.Button(controls, text="Trace uploaded image", command=self._trace_upload).grid(row=0, column=3, padx=6)

        ttk.Label(controls, text="CLIP retrieve threshold").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        self.clip_threshold = tk.DoubleVar(value=0.85)
        ttk.Scale(controls, from_=0.50, to=0.99, variable=self.clip_threshold, orient="horizontal", length=220).grid(
            row=1, column=1, padx=4, pady=4, sticky="ew"
        )
        self.clip_label = ttk.Label(controls, text="0.85")
        self.clip_label.grid(row=1, column=2, sticky="w")
        self.clip_threshold.trace_add("write", lambda *_: self.clip_label.config(text=f"{self.clip_threshold.get():.2f}"))

        ttk.Label(controls, text="pHash max distance").grid(row=1, column=3, padx=4, pady=4, sticky="e")
        self.phash_threshold = tk.IntVar(value=5)
        ttk.Spinbox(controls, from_=0, to=20, textvariable=self.phash_threshold, width=6).grid(row=1, column=4, padx=4)

        self.reason_var = tk.StringVar(value="Add users and posts, or load the sample network, then trace a post.")
        ttk.Label(frame, textvariable=self.reason_var, wraplength=1180, justify="left").pack(anchor="w", pady=(8, 6))

        body = ttk.Panedwindow(frame, orient="horizontal")
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=1)

        cols = ("post", "user", "clip", "phash", "combined", "time")
        self.results_tree = ttk.Treeview(left, columns=cols, show="headings")
        headings = {
            "post": "Post",
            "user": "User",
            "clip": "CLIP",
            "phash": "pHash dist",
            "combined": "Score",
            "time": "Timestamp",
        }
        for col in cols:
            self.results_tree.heading(col, text=headings[col])
            self.results_tree.column(col, width=110 if col != "time" else 150, anchor="w")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scroll.set)
        self.results_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.results_tree.bind("<<TreeviewSelect>>", self._show_comparison)

        self.compare_frame = ttk.LabelFrame(right, text="Image comparison", padding=8)
        self.compare_frame.pack(fill="both", expand=True)

    def _graph_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Network graph")
        controls = ttk.Frame(frame)
        controls.pack(fill="x")
        ttk.Button(controls, text="Refresh graph", command=self._draw_graph).pack(side="left")
        self.graph_info = ttk.Label(controls, text="")
        self.graph_info.pack(side="left", padx=12)
        self.graph_host = ttk.Frame(frame)
        self.graph_host.pack(fill="both", expand=True, pady=(8, 0))

    def _load_model(self) -> None:
        try:
            self.detector.load_clip()
            self.root.after(0, lambda: self._ready("CLIP ready. Add users and posts, or load the sample network."))
        except Exception as exc:
            self.root.after(0, lambda: self._ready(f"CLIP failed to load: {exc}"))

    def _ready(self, message: str) -> None:
        self._set_busy(False)
        self.status.set(message)
        self.refresh()

    def _set_busy(self, busy: bool) -> None:
        self._working = busy
        cursor = "watch" if busy else ""
        self.root.config(cursor=cursor)
        for child in self.root.winfo_children():
            try:
                child.configure(cursor=cursor)
            except tk.TclError:
                pass

    def _run_bg(self, work: Callable, on_ok: Callable, busy_msg: str) -> None:
        if self._working:
            messagebox.showinfo("Busy", "Wait for the current CLIP job to finish.")
            return
        self._set_busy(True)
        self.status.set(busy_msg)

        def runner() -> None:
            try:
                result = work()
            except Exception as exc:
                self.root.after(0, lambda err=exc: self._bg_fail(err))
                return
            self.root.after(0, lambda: self._bg_ok(on_ok, result))

        threading.Thread(target=runner, daemon=True).start()

    def _bg_ok(self, on_ok: Callable, result) -> None:
        self._set_busy(False)
        on_ok(result)

    def _bg_fail(self, exc: Exception) -> None:
        self._set_busy(False)
        self.status.set(str(exc))
        messagebox.showerror("Task failed", str(exc))

    def _combo_id(self, value: str) -> str:
        if " | " in value:
            return value.split(" | ", 1)[0].strip()
        return value.strip()

    def _user_choices(self):
        return [f"{u['id']} | {u['username']}" for u in self.detector.users()]

    def _post_choices(self):
        return [f"{p['id']} | {p['username']}" for p in self.detector.posts()]

    def _browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.gif"), ("All files", "*.*")],
        )
        if path:
            self.image_path_var.set(path)

    def _parse_timestamp(self) -> Optional[datetime]:
        raw = self.timestamp_entry.get().strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError("Timestamp must be YYYY-MM-DD HH:MM:SS") from exc

    def _add_user(self) -> None:
        try:
            user_id = self.detector.add_user(self.username_entry.get(), self.bio_entry.get())
        except Exception as exc:
            messagebox.showerror("Could not add user", str(exc))
            return
        self.username_entry.delete(0, tk.END)
        self.bio_entry.delete(0, tk.END)
        self.status.set(f"Added user {user_id}")
        self.refresh()

    def _add_post(self) -> None:
        if not self.detector.clip_ready:
            messagebox.showwarning("Please wait", "CLIP is still loading.")
            return
        user_id = self._combo_id(self.post_user_combo.get())
        image_path = self.image_path_var.get().strip()
        caption = self.caption_entry.get()
        if not user_id or not image_path:
            messagebox.showerror("Missing input", "Select a user and an image.")
            return
        try:
            timestamp = self._parse_timestamp()
        except ValueError as exc:
            messagebox.showerror("Invalid timestamp", str(exc))
            return

        def work():
            return self.detector.add_post(user_id, image_path, caption, timestamp=timestamp)

        def done(result):
            post_id, source_id = result
            self.image_path_var.set("")
            self.caption_entry.delete(0, tk.END)
            self.timestamp_entry.delete(0, tk.END)
            if source_id:
                self.status.set(f"Added {post_id} as a near-duplicate / repost of {source_id}")
            else:
                self.status.set(f"Added original post {post_id}")
            self.refresh()

        self._run_bg(work, done, "Encoding image with CLIP...")

    def _add_follow(self) -> None:
        follower = self._combo_id(self.follower_combo.get())
        followee = self._combo_id(self.followee_combo.get())
        try:
            self.detector.follow(follower, followee)
        except Exception as exc:
            messagebox.showerror("Could not add follow", str(exc))
            return
        self.status.set(f"{follower} now follows {followee}")
        self.refresh()

    def _add_like(self) -> None:
        user_id = self._combo_id(self.like_user_combo.get())
        post_id = self._combo_id(self.like_post_combo.get())
        try:
            likes = self.detector.like(user_id, post_id)
        except Exception as exc:
            messagebox.showerror("Could not like post", str(exc))
            return
        self.status.set(f"{user_id} liked {post_id} ({likes} like(s))")
        self.refresh()

    def _load_sample(self) -> None:
        if not self.detector.clip_ready:
            messagebox.showwarning("Please wait", "CLIP is still loading.")
            return
        if any(post["caption"].startswith("[sample]") for post in self.detector.posts()):
            if not messagebox.askyesno("Sample network", "Sample posts already exist. Add another sample set?"):
                return

        def work():
            return self.detector.load_sample_network()

        def done(result):
            posts = result["posts"]
            linked = result["linked"]
            extra = f" Alice's copy linked to {linked}." if linked else " Alice's copy was stored without a strict repost edge."
            self.status.set(
                f"Sample network ready: Bob {posts['original']}, Alice {posts['near']}, Carol {posts['other']}." + extra
            )
            self.refresh()
            self.notebook.select(4)
            self._draw_graph()

        self._run_bg(work, done, "Building sample network and encoding images...")

    def _trace_post(self) -> None:
        post_id = self._combo_id(self.analysis_post_combo.get())
        if not post_id:
            messagebox.showerror("Missing input", "Select a post to trace.")
            return
        self._run_trace(post_id=post_id)

    def _trace_upload(self) -> None:
        if not self.detector.clip_ready:
            messagebox.showwarning("Please wait", "CLIP is still loading.")
            return
        path = filedialog.askopenfilename(
            title="Query image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.gif"), ("All files", "*.*")],
        )
        if path:
            self._run_trace(image_path=path)

    def _run_trace(self, post_id: Optional[str] = None, image_path: Optional[str] = None) -> None:
        clip_cut = self.clip_threshold.get()
        hash_cut = int(self.phash_threshold.get())

        def work():
            return self.detector.trace_source(
                post_id=post_id,
                image_path=image_path,
                clip_threshold=clip_cut,
                phash_threshold=hash_cut,
            )

        def done(result):
            self._show_trace(result, post_id, image_path)

        self._run_bg(work, done, "Tracing source...")

    def _show_trace(self, result, post_id: Optional[str], image_path: Optional[str]) -> None:
        self.last_query_path = image_path
        self.last_query_post = post_id
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        if result.origin:
            origin = result.origin
            self.results_tree.insert(
                "",
                "end",
                values=(
                    f"{origin.post_id} (origin)",
                    origin.username,
                    f"{origin.clip_similarity:.4f}",
                    int(origin.phash_distance),
                    f"{origin.combined:.4f}",
                    origin.timestamp.isoformat(sep=" ", timespec="seconds"),
                ),
                tags=("origin",),
            )
        self.results_tree.tag_configure("origin", background="#d7f5d7")

        for match in result.matches:
            if result.origin and match.post_id == result.origin.post_id:
                continue
            self.results_tree.insert(
                "",
                "end",
                values=(
                    match.post_id,
                    match.username,
                    f"{match.clip_similarity:.4f}",
                    int(match.phash_distance),
                    f"{match.combined:.4f}",
                    match.timestamp.isoformat(sep=" ", timespec="seconds"),
                ),
            )

        self.reason_var.set(result.reasoning)
        self.status.set(result.reasoning)
        if result.origin:
            self._render_compare(self._query_image_path(), result.origin.image_path, result.origin.post_id)

    def _query_image_path(self) -> Optional[str]:
        if self.last_query_path:
            return self.last_query_path
        post_id = self.last_query_post
        if post_id and post_id in self.detector.graph:
            return self.detector.graph.nodes[post_id].get("image_path")
        return None

    def _show_comparison(self, _event=None) -> None:
        selection = self.results_tree.selection()
        if not selection:
            return
        post_label = str(self.results_tree.item(selection[0])["values"][0])
        post_id = post_label.replace(" (origin)", "")
        if post_id not in self.detector.graph:
            return
        self._render_compare(self._query_image_path(), self.detector.graph.nodes[post_id]["image_path"], post_id)

    def _render_compare(self, left_path: Optional[str], right_path: Optional[str], right_id: str) -> None:
        for child in self.compare_frame.winfo_children():
            child.destroy()
        self.photo_cache.clear()
        holder = ttk.Frame(self.compare_frame)
        holder.pack(fill="both", expand=True)
        self._thumb_column(holder, "Query", left_path).pack(side="left", expand=True, fill="both", padx=6)
        self._thumb_column(holder, f"Candidate {right_id}", right_path).pack(side="left", expand=True, fill="both", padx=6)

    def _thumb_column(self, parent, title: str, path: Optional[str]) -> ttk.LabelFrame:
        box = ttk.LabelFrame(parent, text=title, padding=6)
        photo = self._load_thumb(path)
        if photo:
            label = ttk.Label(box, image=photo)
            label.image = photo
            label.pack()
            self.photo_cache.append(photo)
        else:
            ttk.Label(box, text="No image").pack()
        if path:
            ttk.Label(box, text=os.path.basename(path), wraplength=220).pack(pady=(6, 0))
        return box

    def _load_thumb(self, path: Optional[str]) -> Optional[ImageTk.PhotoImage]:
        if not path or not os.path.isfile(path):
            return None
        image = Image.open(path).convert("RGB")
        image.thumbnail(THUMB_SIZE)
        return ImageTk.PhotoImage(image)

    def _view_post(self, _event=None) -> None:
        selection = self.posts_tree.selection()
        if not selection:
            return
        post_id = self.posts_tree.item(selection[0])["values"][0]
        data = self.detector.graph.nodes[post_id]
        win = tk.Toplevel(self.root)
        win.title(f"Post {post_id}")
        photo = self._load_thumb(data.get("image_path"))
        if photo:
            label = ttk.Label(win, image=photo)
            label.image = photo
            label.pack(padx=12, pady=12)
            self.photo_cache.append(photo)
        ttk.Label(win, text=f"Caption: {data.get('caption') or '(none)'}").pack(anchor="w", padx=12)
        ttk.Label(win, text=f"Timestamp: {data.get('timestamp')}").pack(anchor="w", padx=12, pady=(0, 12))

    def _draw_graph(self) -> None:
        for child in self.graph_host.winfo_children():
            child.destroy()
        fig = Figure(figsize=(11, 7), dpi=100)
        ax = fig.add_subplot(111)
        graph = self.detector.graph
        users = [n for n, d in graph.nodes(data=True) if d.get("type") == "user"]
        posts = [n for n, d in graph.nodes(data=True) if d.get("type") == "post"]
        if not graph.nodes:
            ax.text(0.5, 0.5, "Graph is empty", ha="center", va="center")
        else:
            pos = nx.spring_layout(graph, seed=7, k=0.85)
            nx.draw_networkx_nodes(graph, pos, nodelist=users, node_color="#8ecae6", node_size=1100, ax=ax)
            nx.draw_networkx_nodes(graph, pos, nodelist=posts, node_color="#b7e4c7", node_size=780, ax=ax)
            posted = [(u, v) for u, v, d in graph.edges(data=True) if d.get("type") == "posted"]
            follows = [(u, v) for u, v, d in graph.edges(data=True) if d.get("type") == "follows"]
            likes = [(u, v) for u, v, d in graph.edges(data=True) if d.get("type") == "likes"]
            reposts = [(u, v) for u, v, d in graph.edges(data=True) if d.get("type") == "reposted_from"]
            nx.draw_networkx_edges(graph, pos, edgelist=posted, edge_color="#2d6a4f", ax=ax, arrows=True)
            nx.draw_networkx_edges(graph, pos, edgelist=follows, edge_color="#1d3557", ax=ax, arrows=True)
            nx.draw_networkx_edges(graph, pos, edgelist=likes, edge_color="#e76f51", ax=ax, arrows=True, style="dotted")
            nx.draw_networkx_edges(graph, pos, edgelist=reposts, edge_color="#9b2226", ax=ax, arrows=True, style="dashed")
            labels = {
                n: d.get("username", n) if d.get("type") == "user" else n
                for n, d in graph.nodes(data=True)
            }
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)
        ax.set_title("Users (blue), posts (green). Dashed red = reposted_from")
        ax.axis("off")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.graph_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.graph_canvas = canvas
        self.graph_info.config(text=f"{len(users)} users, {len(posts)} posts, {graph.number_of_edges()} edges")
        plt.close(fig)

    def _save(self) -> None:
        try:
            self.detector.save()
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.status.set(f"Saved {self.detector.graph_path}")
        messagebox.showinfo("Saved", f"Graph written to {self.detector.graph_path}")

    def refresh(self) -> None:
        for tree in (self.users_tree, self.posts_tree, self.rel_tree):
            for item in tree.get_children():
                tree.delete(item)

        for user in self.detector.users():
            self.users_tree.insert(
                "",
                "end",
                values=(user["id"], user["username"], user["bio"], user["followers"], user["posts"]),
            )
        for post in self.detector.posts():
            self.posts_tree.insert(
                "",
                "end",
                values=(
                    post["id"],
                    f"{post['username']} ({post['user_id']})",
                    post["caption"],
                    post["timestamp"],
                    post["likes"],
                    post["repost_of"] or "",
                ),
            )
        for follower_id, follower_name, followee_id, followee_name in self.detector.follows():
            self.rel_tree.insert(
                "",
                "end",
                values=(f"{follower_name} ({follower_id})", f"{followee_name} ({followee_id})"),
            )

        users = self._user_choices()
        posts = self._post_choices()
        for combo in (
            self.post_user_combo,
            self.follower_combo,
            self.followee_combo,
            self.like_user_combo,
        ):
            combo["values"] = users
        self.like_post_combo["values"] = posts
        self.analysis_post_combo["values"] = posts


def main() -> None:
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except tk.TclError:
        pass
    ImageSourceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
