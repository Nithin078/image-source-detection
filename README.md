# Image Source Detection

A digital-image forensics prototype that answers a simple question:

> Given a suspicious or reposted image, who likely posted the original, and how did copies move through a social network?

The system does **not** rely on metadata alone. Metadata is easy to strip or fake. Instead it stores a visual fingerprint of every uploaded image and models the social graph those images travel through. When a query arrives, it ranks near-duplicates and walks repost edges back to the earliest strong match.

Built during an IFSCR summer internship (project P11, *Image Source Identification in Social-Media-Like Networks*). This repository is the cleaned **CLIP + graph** prototype. Earlier SIFT + histogram drafts were dropped because they do not match the final code.

**Repository:** [github.com/Nithin078/image-source-detection](https://github.com/Nithin078/image-source-detection)

---

## Table of contents

1. [Problem](#problem)
2. [What this project does](#what-this-project-does)
3. [What this project is not](#what-this-project-is-not)
4. [High-level architecture](#high-level-architecture)
5. [How source tracing works](#how-source-tracing-works)
6. [Methods and tools](#methods-and-tools)
7. [Graph model](#graph-model)
8. [Scoring and thresholds](#scoring-and-thresholds)
9. [Choosing the original post](#choosing-the-original-post)
10. [User workflow](#user-workflow)
11. [GUI reference](#gui-reference)
12. [Project layout](#project-layout)
13. [Installation](#installation)
14. [Running the app](#running-the-app)
15. [Optional DGL backend](#optional-dgl-backend)
16. [Tests](#tests)
17. [Data storage](#data-storage)
18. [Limitations](#limitations)
19. [Future work](#future-work)
20. [Resume blurb](#resume-blurb)
21. [License](#license)

---

## Problem

Images move across social networks as screenshots, crops, recompressions, and unattributed reposts. For digital forensics that creates two failures:

- **Content-only matching** can say “these two pictures look alike” but cannot say *who posted first*.
- **Metadata-only analysis** (EXIF camera tags, download timestamps) is unreliable because apps strip or rewrite those fields.

This prototype combines both sides: a visual near-duplicate detector and a small simulated social network. The visual layer finds the cluster of matching posts. The graph + timestamps recover the likely original poster.

---

## What this project does

- Builds a **directed social graph** of users and posts (`follows`, `posted`, `likes`, `reposted_from`).
- Fingerprints every uploaded image with:
  - **CLIP ViT-B/32** image embeddings (semantic / visual similarity)
  - **perceptual hash (pHash)** (near-duplicate / screenshot detection)
- When a new post looks like an older one, writes a **`reposted_from` edge** to the earliest strong match.
- Traces a selected post or an uploaded query image back to the **likely original post and poster**.
- Ranks matches by a combined CLIP + pHash score so stronger provenance evidence is first.
- Uses **follow-graph PageRank only as an “amplifier” hint**, not as the origin.
- Provides a **Tkinter GUI** to post, follow, like, delete, reset, visualize the graph, and inspect side-by-side images.
- Uses **EXIF DateTimeOriginal** (then file mtime) when the operator does not type a timestamp.

It is a **desktop prototype** for a simulated network, not a crawler for Twitter / Instagram / WhatsApp.

---

## What this project is not

- Not a production forensic lab tool and not a court-ready provenance system.
- Not a web app or API server.
- Not a live social-media scraper.
- Not the older SIFT + color-histogram experiment from early internship drafts.
- Not a claim that CLIP cosine *is* a probability of “same photo.” Scores are ranked evidence, not calibrated confidence.

---

## High-level architecture

```text
                    ┌──────────────────────────┐
  image file  ───►  │  Fingerprint             │
                    │  CLIP embedding (L2)     │
                    │  pHash (64-bit)          │
                    │  timestamp (EXIF/mtime)  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  NetworkX MultiDiGraph   │
                    │  users, posts, edges     │
                    │  embeddings/*.npy        │
                    │  hashes.json             │
                    └────────────┬─────────────┘
                                 │
          query post or file ───►│  retrieve near-matches
                                 │  walk / reconstruct path
                                 ▼
                    ┌──────────────────────────┐
                    │  Trace result            │
                    │  origin + ranked matches │
                    │  path + amplifier hint   │
                    └──────────────────────────┘
```

| Layer | File | Role |
|---|---|---|
| GUI | `app.py` | Tabs for users, posts, relationships, tracing, graph. CLIP work runs on a background thread. |
| Core tracer | `detector.py` | Fingerprints, graph mutations, retrieve / link tests, origin selection, JSON+NPY persistence. |
| Optional backend | `dgl_backend.py` | Same CLIP lookup on a DGL heterograph. Not used by the GUI. |
| Tests | `tests/test_detector.py` | Scoring, origin fallback, likes, delete/reset, persistence. No CLIP download. |

---

## How source tracing works

End-to-end, one query goes through these stages.

### 1. Fingerprint the query

For a stored post, the system already has a CLIP vector and a pHash. For an uploaded file it computes both on the spot:

1. Open the image as RGB.
2. Run **CLIP ViT-B/32** (`openai/clip-vit-base-patch32`) and **L2-normalize** the embedding so cosine similarity is a true inner product.
3. Compute a **64-bit pHash** with `imagehash.phash`.
4. If no manual timestamp was given, read **EXIF DateTimeOriginal**, then DateTimeDigitized / DateTime, then the file’s last-modified time.

### 2. Retrieve candidate near-matches

Every stored post is compared to the query:

| Signal | Meaning | Default keep rule |
|---|---|---|
| CLIP cosine | Visual / semantic similarity in \([-1, 1]\) | cosine ≥ **0.85** |
| pHash Hamming distance | Bit flips between 64-bit hashes | distance ≤ **5** |

A post is a **retrieve candidate** if **either** test passes. That is intentional: screenshots of the same photo often have pHash 0, while a crop or filter may survive on CLIP even if the hash drifts.

The GUI slider changes only the CLIP retrieve cutoff (0.50–0.99). The pHash max distance is a spinbox (0–20).

### 3. Score and rank

Each candidate gets a combined score:

```text
combined = 0.7 * clip_cosine + 0.3 * (1 - hamming / 64)
```

CLIP is weighted higher because it survives more edits. pHash is the cheap near-duplicate vote. Results are sorted by combined score, then CLIP, then smaller hash distance.

### 4. Auto-link a repost edge (on ingest, not only on trace)

When **adding** a post, the same comparison runs against earlier posts. A `reposted_from` edge is written only if the match is **stricter** than retrieve:

- CLIP ≥ **0.90** and pHash ≤ **5**, or
- combined score ≥ **0.88**

and the other post is **earlier**. The edge points from the new post to the **earliest** linkable original, not to the most similar recent copy. That keeps chains pointing at the source instead of a mid-path viral account.

Retrieve is allowed to be loose. Linking is not. A lucky pHash collision should not invent fake provenance.

### 5. Recover the origin

For a query post `Q`:

1. Collect retrieve matches.
2. Prefer **link-quality** matches as origin candidates; if none exist, fall back to retrieve-quality matches. `Q` itself is always a candidate.
3. Sort candidates by:
   1. **earliest timestamp**
   2. **shortest path** on the `reposted_from` graph
   3. **highest combined score**
4. If a stored repost chain already walks from `Q` back to that origin, use that path.
5. If no edge was stored (posts added in reverse order, or just under the link cutoff), still pick the earliest strong match and mark the result as an **edge fallback**.

Follow-graph **PageRank is not used to pick the origin**. A later popular account that many people follow would otherwise look like the source. PageRank is reported separately as “likely amplifier.”

### 6. Show evidence

The GUI lists the origin first (green row), then other matches with CLIP / pHash / combined / time. It shows the two images side by side. The network graph highlights:

| Color | Meaning |
|---|---|
| Gold | Chosen origin post |
| Orange | Intermediate posts on the path |
| Tomato | Query post |
| Green | Other posts |
| Blue | Users |
| Dashed red | `reposted_from` edges |
| Thick gold | Path edges |

---

## Methods and tools

### CLIP (ViT-B/32)

- **Library:** Hugging Face `transformers` + `torch`
- **Model:** `openai/clip-vit-base-patch32`
- **Why:** CLIP embeddings stay close for screenshots, mild crops, and recompression — the edits that show up in social reposts — without needing a custom trained matcher.
- **How:** `CLIPModel.get_image_features`, then divide by the L2 norm. Cosine similarity is `dot(a, b)` after that normalization.
- **Device:** CUDA if PyTorch sees a GPU, otherwise CPU. First launch downloads the weights from Hugging Face (on the order of hundreds of MB). CPU load can take one to two minutes.

CLIP is **not** used here as a text–image model. Only the image tower is used.

### Perceptual hash (pHash)

- **Library:** `ImageHash` (`imagehash.phash`)
- **Why:** Extremely cheap near-duplicate test. A screenshot of the same photo often has Hamming distance 0. Unrelated photos sit far away (in a live test, ~31 / 64).
- **How:** 64-bit hash. Distance is the Hamming distance. Similarity used in the combined score is `1 - distance / 64`.

pHash alone is brittle on heavy crops. That is why it is OR-ed with CLIP for retrieval and AND-ed (or combined-score gated) for linking.

### NetworkX MultiDiGraph

- **Library:** `networkx`
- **Why:** The social context is a labeled directed graph. A `MultiDiGraph` is required so a user can both **post** and **like** the same post. A plain `DiGraph` would overwrite the `posted` edge with `likes`.

### PageRank (amplifier only)

- **Library:** `networkx.pagerank`
- **Where:** Follow subgraph among users who posted matching images.
- **Why not origin:** Incoming follow edges measure popularity, not first publication. The report labels it “likely amplifier.”

### Timestamps

Priority when the operator leaves the timestamp field blank:

1. EXIF `DateTimeOriginal` (tag 36867)
2. EXIF `DateTimeDigitized` (36868)
3. EXIF `DateTime` (306)
4. Filesystem mtime
5. Clock time as a last resort

A typed `YYYY-MM-DD HH:MM:SS` always wins and is stored as `timestamp_source = manual`.

### GUI and visualization

- **Tkinter / ttk** for the desktop UI
- **Pillow** for thumbs, sample-image generation, EXIF
- **Matplotlib + NetworkX** for the spring-layout graph, redrawn after every mutation

CLIP encode, sample-network build, and upload-trace run on a **worker thread** so the window does not freeze. Graph drawing stays on the UI thread.

### Persistence

| File | Contents |
|---|---|
| `data/graph.json` | Nodes and typed edges |
| `data/hashes.json` | Post ID → pHash hex |
| `data/embeddings/<post_id>.npy` | L2-normalized CLIP vector |
| `data/stored_images/` | Copy of each posted file |
| `data/sample_images/` | Generated demo images (not personal photos) |

An old `image_network.pkl` from earlier drafts is imported once and rewritten into this format.

### What was rejected

OpenCV SIFT, color histograms, and “temporal graphs” appear in older internship posters. They are **not** in this repo. The live prototype is CLIP + pHash + NetworkX.

---

## Graph model

```text
  alice ──follows──► bob
    │                 │
    │ posted          │ posted
    ▼                 ▼
  P_repost ──reposted_from──► P_orig
    ▲
    │ likes
  carol
```

| Node type | Attributes |
|---|---|
| `user` | `username`, `bio`, `created_at` |
| `post` | `image_path`, `caption`, `timestamp`, `timestamp_source`, `likes` |

| Edge type | Direction | Meaning |
|---|---|---|
| `follows` | follower → followee | Social follow |
| `posted` | user → post | Authorship |
| `likes` | user → post | Like (separate multi-edge from `posted`) |
| `reposted_from` | later post → earlier post | Near-duplicate provenance |

IDs are generated as `U` / `P` plus 8 hex characters. You do not type them unless you are calling the Python API.

---

## Scoring and thresholds

```text
clip_cosine(a, b) = dot(a, b)          # after L2-normalization
phash_sim         = 1 - hamming / 64
combined          = 0.7 * clip_cosine + 0.3 * phash_sim
```

| Gate | Default | Used for |
|---|---|---|
| CLIP retrieve | `0.85` | Candidate list (OR with pHash) |
| pHash retrieve / link distance | `5` | Candidate list and strict link |
| CLIP link | `0.90` | Writing `reposted_from` (AND with pHash) |
| Combined link | `0.88` | Alternate way to write `reposted_from` |

**Worked example from a live run on this machine**

A WhatsApp photo vs a screenshot of that same photo:

| Pair | CLIP cosine | pHash distance | Combined | Decision |
|---|---:|---:|---:|---|
| original vs screenshot | 0.947 | 0 | 0.963 | retrieve + auto-link |
| either vs an unrelated later photo | ~0.42 | 31 | ~0.45 | ignored |

Constants live at the top of `detector.py`:

```python
DEFAULT_CLIP_THRESHOLD = 0.85
DEFAULT_LINK_CLIP_THRESHOLD = 0.90
DEFAULT_PHASH_THRESHOLD = 5
DEFAULT_LINK_COMBINED_THRESHOLD = 0.88
```

---

## Choosing the original post

In words: **earliest strong visual match**, then **shortest stored repost path**.

```text
candidates = link-quality matches, else retrieve matches
candidates += the query post itself

sort by (timestamp ASC, path_length ASC, combined DESC)
origin = first candidate

if a reposted_from walk reaches that origin:
    path = that walk
else:
    path = [origin, query]   # fallback, no stored edge
```

PageRank on `{users who posted a match}` is attached as `centrality_username` only.

---

## User workflow

### Fast demo (recommended)

1. Install and run `python app.py` (see [Installation](#installation)).
2. Wait until the status bar says **CLIP ready**. First start downloads the model.
3. Click **Load sample network**.
   - **Bob** posts a generated “ORIGINAL” image, timestamped 2 hours ago.
   - **Alice** follows Bob and posts a cropped / blurred copy, 1 hour ago.
   - **Carol** posts an unrelated red image.
4. Open **Trace source**, select Alice’s post, click **Trace selected post**.
5. Expected result: origin is Bob’s earlier post, path `Bob's post → Alice's post`.
6. Open **Network graph**. Gold / orange / red should mark that path.

The sample images are generated in `data/sample_images/`. No personal photos are committed to git.

### Manual forensics-style demo

1. **Users** — add `bob` and `alice` (and anyone else).
2. **Posts** — select Bob, browse to a real photo, leave timestamp blank (EXIF / file time) or type an earlier `YYYY-MM-DD HH:MM:SS`.
3. **Relationships** — Alice follows Bob (optional, only needed for the amplifier hint).
4. **Posts** — Alice uploads a screenshot, crop, or recompressed copy of Bob’s image.
5. If the pair is strong enough, the posts table shows Alice’s row with **Repost of** = Bob’s post ID.
6. **Trace source** — select Alice’s post, or use **Trace uploaded image** with a third copy that was never posted.
7. Inspect CLIP / pHash / combined scores and the side-by-side preview.
8. Check the graph highlight.

### Cleanup

- **Delete selected user** removes the user and every post they made (plus stored files and embeddings).
- **Delete selected post** removes one post and its fingerprint.
- **Unfollow** / **Unlike** remove those edges.
- **Reset database** wipes the whole store. The repo itself is untouched.

---

## GUI reference

| Tab | What you do |
|---|---|
| **Users** | Add a username + optional bio. Delete the selected user. |
| **Posts** | Choose a poster, browse an image, optional caption and timestamp. Delete the selected post. Double-click a row to preview. |
| **Relationships** | Add follow, unfollow selected row, like / unlike. |
| **Trace source** | Pick a post or upload a query file. Adjust CLIP retrieve threshold and pHash max distance. |
| **Network graph** | Auto-refreshes after every change. **Clear path highlight** removes gold/orange/red. |

Header buttons: **Load sample network**, **Reset database**. Status bar: current action + **Save database** (saves also happen automatically after each mutation).

CLIP encode, sample load, and upload-trace show a wait cursor. A second CLIP job is blocked until the first finishes.

---

## Project layout

```text
image-source-detection/
├── app.py                 # Tkinter GUI
├── detector.py            # CLIP + pHash + NetworkX tracer
├── dgl_backend.py         # optional DGL experiment
├── requirements.txt
├── LICENSE                # MIT
├── README.md
├── tests/
│   └── test_detector.py
└── data/                  # created at runtime, mostly gitignored
    ├── graph.json
    ├── hashes.json
    ├── embeddings/
    ├── stored_images/
    └── sample_images/
```

---

## Installation

### Requirements

- **Python 3.10+** (developed on 3.12)
- Internet on **first** CLIP download
- A desktop session (Tkinter). This is not a headless server app.
- Optional NVIDIA GPU + CUDA PyTorch build. CPU works; it is slower.

### Windows (PowerShell)

```powershell
git clone https://github.com/Nithin078/image-source-detection.git
cd image-source-detection

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If script activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

or call `.\.venv\Scripts\python.exe app.py` without activating.

### macOS / Linux

```bash
git clone https://github.com/Nithin078/image-source-detection.git
cd image-source-detection

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Debian/Ubuntu you may also need:

```bash
sudo apt install python3-tk python3-venv
```

### What gets installed

| Package | Used for |
|---|---|
| `torch` | CLIP inference |
| `transformers` | CLIP model + processor |
| `Pillow` | Images, EXIF, sample generation |
| `ImageHash` | pHash |
| `networkx` | Social graph + PageRank |
| `matplotlib` | Graph drawing |
| `numpy` | Embeddings and scoring |

`dgl` is **not** in `requirements.txt`. The main app does not import it.

---

## Running the app

Always start from the project root so `data/` lands next to the code:

```bash
python app.py
```

First run:

1. Status bar shows **Loading CLIP (ViT-B/32)...**
2. Hugging Face may download `openai/clip-vit-base-patch32`.
3. Status changes to **CLIP ready**. Then add users or load the sample network.

If CLIP fails (no network, incompatible torch), you can still add users and follows, but posting and tracing uploaded images will refuse until the model loads.

### Python API (no GUI)

```python
from detector import ImageSourceDetector

det = ImageSourceDetector()          # data/ next to detector.py
det.load_clip()

bob = det.add_user("bob")
alice = det.add_user("alice")
det.follow(alice, bob)

p1, _ = det.add_post(bob, "original.jpg")
p2, linked = det.add_post(alice, "screenshot.png")

result = det.trace_source(post_id=p2)
print(result.origin.username, result.origin.clip_similarity, result.path)
print(result.reasoning)
```

Useful methods: `users`, `posts`, `follows`, `like`, `unlike`, `unfollow`, `delete_post`, `delete_user`, `reset`, `load_sample_network`, `find_similar`, `trace_source`.

---

## Optional DGL backend

`dgl_backend.py` rebuilds the internship DGL experiment: a heterogeneous graph with `user/follows/user`, `user/posted/post`, and `post/posted_by/user`, plus CLIP nearest-neighbor lookup. The GUI never imports this file.

```bash
pip install dgl
python dgl_backend.py path/to/original.jpg path/to/query.jpg
```

DGL wheels are OS- and torch-version specific. On some Windows setups the package imports and then fails on a missing `graphbolt` DLL. That error is caught; it cannot crash `app.py`. If you do not need the paper’s DGL mention, skip this file.

---

## Tests

```bash
python -m unittest tests.test_detector -v
```

These tests **do not** download CLIP. They cover:

- cosine / pHash / combined scoring
- retrieve vs link gates
- origin walk when a `reposted_from` edge exists
- origin fallback when the edge is missing
- origin prefers earliest match over PageRank
- author can like their own post without losing `posted`
- JSON + `.npy` round-trip
- EXIF string parse and mtime fallback
- delete post, delete user, unfollow, unlike, reset

---

## Data storage

Runtime files under `data/` are gitignored except `.gitkeep` placeholders. After **Reset database**, `graph.json` is an empty graph; stored images and embeddings are deleted; sample images can be regenerated.

If you copy the project, do **not** commit `data/stored_images/` if you posted personal photos.

To wipe everything by hand instead of the GUI:

```text
delete data/graph.json
delete data/hashes.json
delete data/embeddings/*.npy
delete data/stored_images/*   (keep .gitkeep)
```

---

## Limitations

- The network is **simulated**. There is no connector to a real platform.
- Matching is **linear** over all stored embeddings. Fine for a demo, not for millions of posts (that would need FAISS / HNSW).
- CLIP cosine is **not** a calibrated probability of “same photograph.” Two different photos of the same landmark can score high.
- pHash can collide or fail on strong crops, overlays, or memes with added text.
- Origin quality depends on **who posted first in this graph**. If the true first copy was never ingested, the system reports the earliest copy it has.
- Combined score weights (`0.7 / 0.3`) and the numeric cutoffs are prototype defaults, not fitted on a labeled forensic dataset.
- The GUI is single-operator and local.

Treat every result as a **ranked hypothesis** for an investigator, not as a ground-truth source.

---

## Future work

Reasonable extensions that are **out of scope** for this repo:

- Approximate nearest neighbor index for large galleries
- Video keyframe provenance
- Fit retrieve/link thresholds on a labeled near-duplicate set
- Export a written trace report (JSON / PDF)
- Multi-user or server deployment

Do not put SIFT back unless you are comparing methods in a paper.

---

## Resume blurb

During my IFSCR summer internship I built an image source-detection prototype for digital image forensics. The system models a social network of users and posts (NetworkX / DGL) and stores visual fingerprints of uploaded images using CLIP embeddings and perceptual hashes. When a suspicious or reposted image is submitted, it finds near-matches by cosine similarity / hash distance and walks repost edges on the graph to recover the likely original post and poster. Matches are ranked by similarity confidence so investigators can prioritize the strongest provenance signal. A small GUI was used to demo posting, following, and source tracing end-to-end.

---

## License

MIT. See [LICENSE](LICENSE). Copyright (c) 2025 R Nithin Kumar Reddy.
