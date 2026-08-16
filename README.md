# Image Source Detection

A digital-image forensics prototype that finds the likely original poster of a reposted image.

The system stores visual fingerprints of every uploaded image and models the social network those images move through. When a suspicious or reposted image is submitted, it ranks near-matches and walks repost edges to recover the earliest post and poster.

Built during an IFSCR summer internship. This is the cleaned CLIP + graph prototype, not the earlier SIFT experiment.

## What it does

- **CLIP (ViT-B/32)** image embeddings with L2-normalized cosine similarity
- **pHash** near-duplicate / repost detection (Hamming distance)
- **NetworkX** social graph: user, post, follows, likes, `reposted_from`
- Source tracing with a ranked similarity score
- Optional **DGL** heterograph backend for the same lookup
- Tkinter GUI to demo posting, following, liking, and tracing end to end

## How source tracing works

1. Fingerprint the query with a CLIP embedding and a perceptual hash.
2. Rank stored posts by a combined score: `0.7 * cosine + 0.3 * (1 - hamming/64)`.
3. Keep posts above the CLIP threshold **or** inside the pHash distance cutoff (defaults `0.90` / `5`).
4. If the query is an existing post, walk `reposted_from` edges back to the origin. Otherwise pick the earliest near-match.
5. PageRank on the follow subgraph of matching posters is reported as a secondary social signal.

Matches are ranked so the strongest provenance signal is first.

## Project layout

```text
image-source-detection/
  app.py              GUI
  detector.py         CLIP + pHash + NetworkX tracer
  dgl_backend.py      optional DGL experiment
  tests/              scoring and graph-walk tests (no GPU needed)
  data/               runtime database and stored image copies
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

1. Add two users, for example `alice` and `bob`.
2. Have `bob` post an original image.
3. Have `alice` follow `bob`, then post a crop, screenshot, or recompressed copy of the same image.
4. Open **Trace source**, select Alice's post (or upload the copy), and run the trace.
5. The origin should be Bob's earlier post, with CLIP / pHash scores and the repost path.
6. **Network graph** shows users in blue, posts in green, and dashed red `reposted_from` edges.

The graph and embeddings are saved automatically to `data/image_network.pkl`. Stored copies live in `data/stored_images/`. Both are gitignored.

## Tests

```bash
python -m unittest discover -s tests -v
```

These tests cover cosine / pHash scoring and origin walking. They do not download CLIP.

## Resume blurb

During my IFSCR summer internship I built an image source-detection prototype for digital image forensics. The system models a social network of users and posts (NetworkX / DGL) and stores visual fingerprints of uploaded images using CLIP embeddings and perceptual hashes. When a suspicious or reposted image is submitted, it finds near-matches by cosine similarity / hash distance and walks repost edges on the graph to recover the likely original post and poster. Matches are ranked by similarity confidence so investigators can prioritize the strongest provenance signal. A small GUI was used to demo posting, following, and source tracing end-to-end.
