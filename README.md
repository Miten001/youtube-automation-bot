# Video Uploader — Auto-tag, Enhance & Upload to YouTube

Automate publishing videos to **YouTube**. For every video the tool will:

1. **Enhance quality** — re-encode with ffmpeg (upscale/denoise/sharpen, high-quality
   x264, audio loudness normalisation, web-optimised `faststart`).
2. **Auto-generate tags** — extract relevant keywords from the title, description
   and filename, and add USA-targeted hashtags (before upload).
3. **Upload** — resumable upload to YouTube (Data API v3) with automatic retries.

---

## 1. Prerequisites

- **Python 3.10+**
- **ffmpeg** (required for quality enhancement):
  - Ubuntu/Debian: `sudo apt-get install -y ffmpeg`
  - Fedora: `sudo dnf install -y ffmpeg`
  - macOS: `brew install ffmpeg`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Configure credentials

Copy the examples and fill in your values:

```bash
cp .env.example .env
cp config.example.yaml config.yaml
```

### YouTube (OAuth 2.0)
1. Go to the [Google Cloud Console](https://console.cloud.google.com/), create a project.
2. Enable the **YouTube Data API v3**.
3. Create an **OAuth client ID** of type **Desktop app** and download the JSON.
4. Save it as `client_secret.json` (or point `YT_CLIENT_SECRET_FILE` to it in `.env`).
5. On the first upload a browser opens for consent; the token is cached in `token.json`.

---

## 3. Usage

Preview the tags a video would get (no upload):

```bash
python -m src.main tags --title "How to Bake Bread" --description "easy beginner recipe"
```

Upload a single video (enhanced + auto-tagged):

```bash
python -m src.main -c config.yaml upload video.mp4 \
    --title "My Awesome Video" \
    --description "Full description here"
```

**No title? No problem.** Omit `--title` and the tool auto-generates a catchy
title for the video from its description / keywords / filename:

```bash
python -m src.main upload morning_home_workout.mp4
# -> auto title: "This Morning Home Workout Is INSANE 💪 #usa #routine"
```

Preview an auto-generated title without uploading:

```bash
python -m src.main title --video epic_minecraft_gameplay.mp4
python -m src.main title --description "best street food tour" --style question
python -m src.main title --hint "homemade coffee" --style clean
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--title "..."` | Set the title (omit to auto-generate) |
| `--tags a,b,c` | Use these tags instead of auto-generating |
| `--no-enhance` | Skip ffmpeg quality enhancement |
| `--thumbnail img.jpg` | Custom YouTube thumbnail |
| `--quiet` | Less logging |

Bulk upload many videos from a manifest (`jobs.example.json` shows the format):

```bash
python -m src.main batch jobs.json
```

---

## 4. Tuning quality & tags

Edit `config.yaml`:

- `enhance.target_height` — output resolution (1080, 1440, 2160, or `null` to keep original).
- `enhance.crf` — lower = higher quality/larger file (18–23 is a good range).
- `enhance.apply_filters` — toggle denoise/sharpen/contrast pass.
- `tags.max_tags`, `tags.always_include`, `tags.stopwords_extra`.
- `tags.region_hashtags` — USA-targeted hashtags merged into every video's tag list
  (default: `usa`, `america`, `trendingusa`, `viralusa`, `fypusa`). Set to `[]` to disable.
- `title_hashtags.enabled` / `title_hashtags.count` — append hashtags to the title.
- `youtube.privacy_status` — `public` / `unlisted` / `private`.

### Auto title generation

If you don't pass `--title`, a catchy title is generated automatically:

- The **topic** is derived from the description, then content keywords, then a
  cleaned filename (camera junk like `VID_`, dates and numbers are ignored).
- It's dropped into an engaging template chosen by `auto_title.style`
  (`catchy` / `question` / `clean`) and a relevant emoji is added.
- The choice is **stable per video** (same file → same title).
- Hashtags are then appended on top (see below), kept under YouTube's 100-char limit.

```bash
python -m src.main title --video epic_minecraft_gameplay.mp4
# title:          Watch This Epic Minecraft Gameplay Till The End 🎮
# title+hashtags: Watch This Epic Minecraft Gameplay Till The End 🎮 #usa
```

### Hashtags in title + USA tags — how it works

- **Title:** one USA hashtag plus the top content hashtags are appended, e.g.
  `Morning Workout Routine #usa #morningworkout #workoutroutine` (kept under YouTube's
  100-char title limit).
- **Tags:** USA-based hashtags are placed first, followed by the auto-generated content
  tags, e.g. `usa, america, trendingusa, viralusa, fypusa, workout, morning, ...`.

Preview both without uploading:

```bash
python -m src.main tags --title "Morning Workout Routine" \
    --description "10 minute full body home workout no equipment"
# title: Morning Workout Routine #usa #morningworkout #workoutroutine
# tags:  usa, america, trendingusa, viralusa, fypusa, workout, morning, routine, ...
```

---

## 5. Project layout

```
video-uploader/
├── src/
│   ├── config.py             # config + secrets loading
│   ├── enhancer.py           # ffmpeg quality enhancement
│   ├── tagger.py             # auto tag generation
│   ├── title_generator.py    # auto catchy title generation
│   ├── youtube_uploader.py   # YouTube Data API v3
│   ├── pipeline.py           # enhance -> tag -> upload orchestration
│   └── main.py               # CLI
├── requirements.txt
├── config.example.yaml
├── .env.example
└── jobs.example.json
```

## Notes
- Enhancement runs first and the **enhanced** file is what gets uploaded. If
  enhancement fails, the pipeline falls back to uploading the original and records
  the error.
- Secrets (`.env`, `token.json`, `client_secret.json`, `config.yaml`) are gitignored.
