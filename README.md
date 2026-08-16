# Image Source Detection

A digital-image forensics prototype that finds the likely original poster of a reposted image.

The system stores visual fingerprints of every uploaded image and models the social network those images move through. When a suspicious or reposted image is submitted, it ranks near-matches and recovers the earliest strong post and poster.

Built during an IFSCR summer internship. This is the cleaned CLIP + graph prototype, not the earlier SIFT experiment.

## What it does

- **CLIP (ViT-B/32)** image embeddings with L2-normalized cosine similarity
- **pHash** near-duplicate / repost detection (Hamming distance)
- **NetworkX MultiDiGraph** social graph: user, post, follows, likes, `reposted_from`
- Source tracing by earliest strong match, then shortest repost path
- Follow-graph **PageRank as an amplifier hint**, not as the origin
- Optional **DGL** heterograph backend for the same lookup
- Tkinter GUI to demo posting, following, liking, tracing, deleting, and reset
- Blank post timestamps use **EXIF DateTimeOriginal**, then file time
- After a trace, the network graph highlights the origin path

## How source tracing works

1. Fingerprint the query with a CLIP embedding and a perceptual hash.
2. Rank stored posts by a combined score: `0.7 * cosine + 0.3 * (1 - hamming/64)`.
3. **Retrieve** candidates if CLIP cosine ≥ `0.85` **or** pHash distance ≤ `5`.
4. **Auto-link** a `reposted_from` edge only if the match is strict: CLIP ≥ `0.90` and pHash ≤ `5`, or combined score ≥ `0.88`.
5. Pick origin as the earliest strong match. If a repost chain exists, prefer the shortest path on that chain. If the edge was never stored, fall back to the earliest strong near-match anyway.
6. Report follow-graph PageRank as “likely amplifier,” not as the source.

## Project layout

```text
image-source-detection/
  app.py              GUI
  detector.py         CLIP + pHash + NetworkX tracer
  dgl_backend.py      optional DGL experiment
  tests/              scoring, graph-walk, and persistence tests
  data/               runtime store (gitignored except placeholders)
    graph.json
    hashes.json
    embeddings/*.npy
    stored_images/
    sample_images/
```

Earlier internship drafts used SIFT + histograms + temporal graphs. That path was dropped. The version in this repo is CLIP + graph.

## Setup

Python 3.10+ recommended. First launch downloads `openai/clip-vit-base-patch32` from Hugging Face.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

DGL is optional and is not in `requirements.txt` because the Windows install is environment-specific. If you want the heterograph demo:

```bash
pip install dgl
python dgl_backend.py path/to/original.jpg path/to/query.jpg
```

## Demo flow

Fastest path: click **Load sample network**. That creates Bob (original, 2 hours ago), Alice (crop / blur of Bob’s image, 1 hour ago), and Carol (unrelated), then encodes the images in a background thread.

Manual path:

1. Add two users, for example `alice` and `bob`.
2. Have `bob` post an original image. You can backdate it with `YYYY-MM-DD HH:MM:SS`.
3. Have `alice` follow `bob`, then post a crop, screenshot, or recompressed copy.
4. Open **Trace source**, select Alice's post (or upload the copy), and run the trace.
5. The origin should be Bob's earlier post, with CLIP / pHash scores and the repost path.
6. **Network graph** updates after every change. After a trace, gold is the origin, orange is the path, and red is the query post.

If you leave the timestamp blank, the post uses the photo's EXIF date, or the file time if EXIF is missing. Use **Delete selected** / **Unfollow** / **Unlike** to edit the graph, or **Reset database** to wipe the store without deleting the repo.

Adding a post or tracing an uploaded image runs CLIP off the UI thread so the window stays responsive.

The graph is saved as `data/graph.json`, hashes as `data/hashes.json`, and CLIP vectors as `data/embeddings/*.npy`. Stored copies live in `data/stored_images/`. An old `image_network.pkl` is imported once and rewritten into this format.

## Tests

```bash
python -m unittest tests.test_detector -v
```

These tests cover scoring, origin fallback, likes-on-own-post, PageRank-as-amplifier, and JSON/NPY persistence. They do not download CLIP.

## Resume blurb

During my IFSCR summer internship I built an image source-detection prototype for digital image forensics. The system models a social network of users and posts (NetworkX / DGL) and stores visual fingerprints of uploaded images using CLIP embeddings and perceptual hashes. When a suspicious or reposted image is submitted, it finds near-matches by cosine similarity / hash distance and walks repost edges on the graph to recover the likely original post and poster. Matches are ranked by similarity confidence so investigators can prioritize the strongest provenance signal. A small GUI was used to demo posting, following, and source tracing end-to-end.
