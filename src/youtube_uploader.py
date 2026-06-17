"""YouTube uploader using the YouTube Data API v3.

Handles OAuth 2.0 (installed/desktop app flow) with a cached refresh token, and
performs a resumable upload so large files survive transient network errors.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
RETRIABLE_STATUS_CODES = {500, 502, 503, 504}
MAX_RETRIES = 8


class YouTubeError(RuntimeError):
    pass


@dataclass
class YouTubeResult:
    video_id: str
    url: str


def _get_credentials(client_secret_file: str, token_file: str) -> Credentials:
    creds: Optional[Credentials] = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(client_secret_file):
                raise YouTubeError(
                    f"Client secret file not found: {client_secret_file}"
                )
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
            # run_local_server opens a browser for consent on first run.
            creds = flow.run_local_server(port=0)
        with open(token_file, "w", encoding="utf-8") as fh:
            fh.write(creds.to_json())

    return creds


def upload_video(
    video_path: str,
    title: str,
    description: str,
    tags: Optional[List[str]],
    category_id: str,
    privacy_status: str,
    client_secret_file: str,
    token_file: str,
    made_for_kids: bool = False,
    thumbnail_path: Optional[str] = None,
) -> YouTubeResult:
    """Upload a video to YouTube and return its id + URL."""
    if not Path(video_path).exists():
        raise YouTubeError(f"Video file not found: {video_path}")

    creds = _get_credentials(client_secret_file, token_file)
    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": title[:100],  # YouTube title hard limit
            "description": description,
            "tags": tags or [],
            "categoryId": str(category_id),
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": made_for_kids,
        },
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(
        part="snippet,status", body=body, media_body=media
    )

    response = _resumable_execute(request)
    video_id = response["id"]

    if thumbnail_path and Path(thumbnail_path).exists():
        try:
            youtube.thumbnails().set(
                videoId=video_id, media_body=MediaFileUpload(thumbnail_path)
            ).execute()
        except HttpError:
            # Thumbnails require a verified channel; ignore if not allowed.
            pass

    return YouTubeResult(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
    )


def _resumable_execute(request):
    """Execute a resumable upload with exponential backoff."""
    response = None
    error = None
    retry = 0
    while response is None:
        try:
            _status, response = request.next_chunk()
            if response is not None and "id" not in response:
                raise YouTubeError(f"Unexpected upload response: {response}")
        except HttpError as exc:
            if exc.resp.status in RETRIABLE_STATUS_CODES:
                error = exc
            else:
                raise YouTubeError(f"YouTube upload failed: {exc}") from exc
        except (OSError, ConnectionError) as exc:
            error = exc

        if error is not None:
            retry += 1
            if retry > MAX_RETRIES:
                raise YouTubeError(f"Exceeded retries; last error: {error}")
            sleep_for = min(2 ** retry + random.random(), 60)
            time.sleep(sleep_for)
            error = None

    return response
