"""Pipeline orchestration: enhance -> tag -> upload to YouTube."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import enhancer, tagger, title_generator, youtube_uploader
from .config import Config


@dataclass
class UploadJob:
    video_path: str
    title: str = ""  # empty -> auto-generate a catchy title
    description: str = ""
    tags: Optional[List[str]] = None  # if None -> auto-generate
    thumbnail_path: Optional[str] = None


@dataclass
class PipelineResult:
    source_video: str
    enhanced_video: Optional[str] = None
    title: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    youtube_url: Optional[str] = None
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, flush=True)


def run_pipeline(
    job: UploadJob,
    config: Config,
    verbose: bool = True,
) -> PipelineResult:
    """Run the full enhance -> tag -> upload pipeline for one video."""
    result = PipelineResult(source_video=job.video_path)

    if not config.youtube.get("enabled", True):
        result.errors["targets"] = "YouTube uploading is disabled in config."
        return result

    config.validate_for_targets()

    # 1) ENHANCE -------------------------------------------------------------
    upload_path = job.video_path
    if config.enhance.get("enabled", True):
        _log(verbose, f"[1/3] Enhancing video quality: {job.video_path}")
        out_dir = Path(config.output_dir)
        out_path = out_dir / f"{Path(job.video_path).stem}_enhanced.mp4"
        try:
            upload_path = enhancer.enhance_video(
                job.video_path, str(out_path), config.enhance
            )
            result.enhanced_video = upload_path
            _log(verbose, f"      -> enhanced file: {upload_path}")
        except enhancer.EnhancerError as exc:
            # Enhancement failure should not block upload of the original.
            result.errors["enhance"] = str(exc)
            _log(verbose, f"      ! enhancement skipped: {exc}")
            upload_path = job.video_path
    else:
        _log(verbose, "[1/3] Enhancement disabled; uploading original.")

    # 2) TAGS ----------------------------------------------------------------
    region_hashtags = config.tags.get("region_hashtags", [])
    max_tags = config.tags.get("max_tags", 15)

    tags = job.tags
    content_tags: List[str] = []
    if tags is None and config.tags.get("enabled", True):
        _log(verbose, "[2/3] Auto-generating tags...")
        content_tags = tagger.generate_tags(
            title=job.title,
            description=job.description,
            filename=job.video_path,
            max_tags=max_tags,
            always_include=config.tags.get("always_include", []),
            extra_stopwords=config.tags.get("stopwords_extra", []),
        )
        tags = content_tags
    else:
        content_tags = list(tags or [])
    tags = tags or []

    # Merge USA-based hashtags into the tag list (region tags first).
    if region_hashtags:
        tags = tagger.add_region_hashtags(tags, region_hashtags, max_tags)
    result.tags = tags
    _log(verbose, f"      -> tags: {', '.join(tags) if tags else '(none)'}")

    # Determine the base title: use the provided one, or auto-generate a catchy
    # one from the description / content keywords / filename.
    at_cfg = config.auto_title
    if job.title:
        base_title = job.title
    elif at_cfg.get("enabled", True):
        base_title = title_generator.generate_title(
            filename=job.video_path,
            description=job.description,
            keywords=content_tags,
            style=at_cfg.get("style", "catchy"),
            max_length=at_cfg.get("max_length", 70),
            emoji=at_cfg.get("emoji", True),
        )
        _log(verbose, f"      -> auto title: {base_title}")
    else:
        base_title = Path(job.video_path).stem

    # Append hashtags to the title: one USA hashtag + top content hashtags.
    final_title = base_title
    th_cfg = config.title_hashtags
    if th_cfg.get("enabled", True):
        final_title = tagger.build_title_with_hashtags(
            title=base_title,
            tags=content_tags,
            region_hashtags=region_hashtags,
            count=th_cfg.get("count", 3),
        )
    _log(verbose, f"      -> title: {final_title}")
    result.title = final_title

    # 3) UPLOAD --------------------------------------------------------------
    _log(verbose, "[3/3] Uploading to YouTube...")
    try:
        yt = youtube_uploader.upload_video(
            video_path=upload_path,
            title=final_title,
            description=job.description,
            tags=tags,
            category_id=config.youtube.get("category_id", "22"),
            privacy_status=config.youtube.get("privacy_status", "private"),
            client_secret_file=config.secrets.yt_client_secret_file,
            token_file=config.secrets.yt_token_file,
            made_for_kids=config.youtube.get("made_for_kids", False),
            thumbnail_path=job.thumbnail_path,
        )
        result.youtube_url = yt.url
        _log(verbose, f"      -> YouTube: {yt.url}")
    except Exception as exc:  # noqa: BLE001 - report failure
        result.errors["youtube"] = str(exc)
        _log(verbose, f"      ! YouTube upload failed: {exc}")

    return result
