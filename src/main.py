"""CLI entrypoint for the video upload automation tool.

Examples
--------
  # Single video to YouTube, auto tags + enhancement
  python -m src.main upload video.mp4 --title "My Video" --description "..."

  # Custom tags (skips auto-tagging)
  python -m src.main upload video.mp4 --title "Demo" --tags python,tutorial,coding

  # Batch: process every video listed in a JSON manifest
  python -m src.main batch jobs.json

  # Preview generated tags without uploading
  python -m src.main tags --title "How to bake bread" --description "easy recipe"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .config import Config
from .pipeline import UploadJob, run_pipeline
from .tagger import add_region_hashtags, build_title_with_hashtags, generate_tags
from .title_generator import generate_title


def _parse_tags(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [t.strip() for t in value.split(",") if t.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-uploader",
        description="Enhance, auto-tag, and upload videos to YouTube.",
    )
    parser.add_argument("-c", "--config", help="Path to YAML config file.")
    sub = parser.add_subparsers(dest="command", required=True)

    # upload -----------------------------------------------------------------
    up = sub.add_parser("upload", help="Upload a single video.")
    up.add_argument("video", help="Path to the source video file.")
    up.add_argument(
        "--title",
        default="",
        help="Video title. Omit to auto-generate a catchy title.",
    )
    up.add_argument("--description", default="", help="Video description.")
    up.add_argument("--tags", help="Comma-separated tags (skips auto-tagging).")
    up.add_argument("--thumbnail", help="Optional thumbnail image (YouTube).")
    up.add_argument("--no-enhance", action="store_true", help="Skip quality enhancement.")
    up.add_argument("--quiet", action="store_true", help="Reduce logging.")

    # batch ------------------------------------------------------------------
    ba = sub.add_parser("batch", help="Upload many videos from a JSON manifest.")
    ba.add_argument("manifest", help="Path to JSON file with a list of jobs.")
    ba.add_argument("--no-enhance", action="store_true")
    ba.add_argument("--quiet", action="store_true")

    # title (preview only) ---------------------------------------------------
    ti = sub.add_parser("title", help="Preview an auto-generated catchy title.")
    ti.add_argument("--video", "--filename", dest="filename", default=None,
                    help="Video file path (used to derive the topic).")
    ti.add_argument("--description", default="", help="Optional description.")
    ti.add_argument("--hint", default=None, help="Optional topic hint.")
    ti.add_argument("--style", default=None,
                    help="catchy | question | clean (default from config).")

    # tags (preview only) ----------------------------------------------------
    tg = sub.add_parser("tags", help="Preview auto-generated tags (no upload).")
    tg.add_argument("--title", default="")
    tg.add_argument("--description", default="")
    tg.add_argument("--filename", default=None)
    tg.add_argument("--max-tags", type=int, default=15)

    return parser


def _apply_overrides(config: Config, args) -> None:
    if getattr(args, "no_enhance", False):
        config.enhance["enabled"] = False


def _print_result(result) -> None:
    print("\n=== Result ===")
    print(f"source:   {result.source_video}")
    if result.enhanced_video:
        print(f"enhanced: {result.enhanced_video}")
    if result.title:
        print(f"title:    {result.title}")
    print(f"tags:     {', '.join(result.tags) if result.tags else '(none)'}")
    print(f"youtube:  {result.youtube_url or '-'}")
    if result.errors:
        print("errors:")
        for key, msg in result.errors.items():
            print(f"  - {key}: {msg}")


def _cmd_upload(args, config: Config) -> int:
    _apply_overrides(config, args)
    job = UploadJob(
        video_path=args.video,
        title=args.title,
        description=args.description,
        tags=_parse_tags(args.tags),
        thumbnail_path=args.thumbnail,
    )
    result = run_pipeline(job, config, verbose=not args.quiet)
    _print_result(result)
    return 0 if result.ok else 1


def _cmd_batch(args, config: Config) -> int:
    _apply_overrides(config, args)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    jobs_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(jobs_data, list):
        print("Manifest must be a JSON array of job objects.", file=sys.stderr)
        return 2

    exit_code = 0
    for i, item in enumerate(jobs_data, 1):
        print(f"\n########## Job {i}/{len(jobs_data)} ##########")
        job = UploadJob(
            video_path=item["video_path"],
            title=item.get("title", ""),
            description=item.get("description", ""),
            tags=item.get("tags"),
            thumbnail_path=item.get("thumbnail_path"),
        )
        result = run_pipeline(job, config, verbose=not args.quiet)
        _print_result(result)
        if not result.ok:
            exit_code = 1
    return exit_code


def _cmd_tags(args, config: Config) -> int:
    region_hashtags = config.tags.get("region_hashtags", [])
    max_tags = config.tags.get("max_tags", 15)
    content_tags = generate_tags(
        title=args.title,
        description=args.description,
        filename=args.filename,
        max_tags=max_tags,
        always_include=config.tags.get("always_include", []),
        extra_stopwords=config.tags.get("stopwords_extra", []),
    )
    tags = content_tags
    if region_hashtags:
        tags = add_region_hashtags(content_tags, region_hashtags, max_tags)

    title = build_title_with_hashtags(
        title=args.title,
        tags=content_tags,
        region_hashtags=region_hashtags,
        count=config.title_hashtags.get("count", 3),
    )
    print(f"title: {title}")
    print(f"tags:  {', '.join(tags) if tags else '(no tags generated)'}")
    return 0


def _cmd_title(args, config: Config) -> int:
    # Derive content keywords first so the title can use them.
    content_tags = generate_tags(
        title="",
        description=args.description,
        filename=args.filename,
        max_tags=config.tags.get("max_tags", 15),
        always_include=config.tags.get("always_include", []),
        extra_stopwords=config.tags.get("stopwords_extra", []),
    )
    at_cfg = config.auto_title
    base_title = generate_title(
        filename=args.filename,
        description=args.description,
        keywords=content_tags,
        hint=args.hint,
        style=args.style or at_cfg.get("style", "catchy"),
        max_length=at_cfg.get("max_length", 70),
        emoji=at_cfg.get("emoji", True),
    )
    region_hashtags = config.tags.get("region_hashtags", [])
    full_title = build_title_with_hashtags(
        title=base_title,
        tags=content_tags,
        region_hashtags=region_hashtags,
        count=config.title_hashtags.get("count", 3),
    )
    print(f"title:           {base_title}")
    print(f"title+hashtags:  {full_title}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = Config.load(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "upload":
            return _cmd_upload(args, config)
        if args.command == "batch":
            return _cmd_batch(args, config)
        if args.command == "title":
            return _cmd_title(args, config)
        if args.command == "tags":
            return _cmd_tags(args, config)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
