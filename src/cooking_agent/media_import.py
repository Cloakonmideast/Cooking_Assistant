"""Get the text of a recipe out of social media posts and audio/video.

* ``YtDlpFetcher`` reads a post's caption/description and subtitles with
  yt-dlp (YouTube, Instagram, TikTok, Facebook, X, Vimeo, Pinterest and
  ~1800 other sites) without downloading the video, and can download just a
  small audio track when the recipe is only spoken.
* Transcribers turn that audio (or a local audio/video file) into text:
  Deepgram's pre-recorded API when ``DEEPGRAM_API_KEY`` is set, else Gemini.

This is for saving recipes for your own cooking. Downloaded audio goes to a
temporary folder that is deleted after transcription.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from pydantic import BaseModel, Field

from .providers.base import LLMProvider, MediaPart

log = logging.getLogger(__name__)

MEDIA_HOSTS = re.compile(
    r"^https?://([a-z0-9-]+\.)*(youtube\.com|youtu\.be|instagram\.com|tiktok\.com|facebook\.com|fb\.watch|"
    r"x\.com|twitter\.com|vimeo\.com|pinterest\.[a-z.]+|pin\.it|reddit\.com|dailymotion\.com|threads\.net)/",
    re.IGNORECASE,
)
AUDIO_VIDEO_SUFFIXES = {".mp3", ".m4a", ".aac", ".wav", ".ogg", ".opus", ".flac", ".mp4", ".mov", ".webm",
                        ".mkv", ".m4v", ".3gp"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".gif"}
INLINE_LIMIT_BYTES = 19 * 1024 * 1024  # Gemini inline request limit is 20 MB


class MediaError(RuntimeError):
    """Could not get anything usable from the media; the message says why."""


def is_media_url(url: str) -> bool:
    return MEDIA_HOSTS.match(url) is not None


def mime_for(path: Path) -> str:
    mt, _ = mimetypes.guess_type(path.name)
    if mt:
        return mt
    return {".m4a": "audio/mp4", ".opus": "audio/ogg", ".heic": "image/heic", ".heif": "image/heif",
            ".mkv": "video/x-matroska"}.get(path.suffix.lower(), "application/octet-stream")


# ------------------------------------------------------------------ captions
def parse_vtt(vtt: str) -> str:
    """WebVTT subtitles -> plain text. Auto-captions repeat each line as it
    scrolls; consecutive duplicates are dropped."""
    out: list[str] = []
    for raw in vtt.splitlines():
        line = raw.strip()
        if (not line or line == "WEBVTT" or "-->" in line or line.isdigit()
                or line.startswith(("Kind:", "Language:", "NOTE", "STYLE"))):
            continue
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line and (not out or out[-1] != line):
            out.append(line)
    return "\n".join(out)


# ------------------------------------------------------------------ fetching
@dataclass
class MediaInfo:
    url: str
    title: str
    caption: str  # post caption / video description, as written
    subtitles: Optional[str] = None  # creator subtitles or auto-captions, as plain text
    uploader: Optional[str] = None


class MediaFetcher(Protocol):
    def info(self, url: str) -> MediaInfo: ...
    def download_audio(self, url: str, dest: Path) -> Path: ...


class _QuietLogger:
    """Route yt-dlp's own output to our log instead of the terminal; errors
    reach the user once, through MediaError."""

    def debug(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)

    info = debug
    warning = debug
    error = debug


class YtDlpFetcher:
    def __init__(self, cookies_from_browser: Optional[str] = None, languages: tuple[str, ...] = ("en",)):
        self.cookies_from_browser = cookies_from_browser or os.environ.get("YTDLP_COOKIES_FROM_BROWSER") or None
        self.languages = languages

    def _opts(self, **extra) -> dict:
        opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "logger": _QuietLogger(), **extra}
        if self.cookies_from_browser:
            opts["cookiesfrombrowser"] = (self.cookies_from_browser,)
        return opts

    def _ydl(self, **extra):
        try:
            import yt_dlp
        except ImportError as e:  # pragma: no cover - dependency is declared
            raise MediaError("yt-dlp is not installed: pip install yt-dlp") from e
        return yt_dlp.YoutubeDL(self._opts(**extra))

    def _wrap(self, url: str, e: Exception) -> MediaError:
        msg = str(e).replace("ERROR: ", "").split("; please report this issue")[0]
        hint = ""
        if "instagram" in url.lower() or "login" in msg.lower() or "cookies" in msg.lower():
            hint = (" This site may need you to be logged in: pass --cookies-from-browser chrome "
                    "(or firefox, safari, edge) to use your browser's login.")
        return MediaError(f"could not read {url}: {msg}.{hint}")

    def info(self, url: str) -> MediaInfo:
        try:
            with self._ydl(skip_download=True) as ydl:
                data = ydl.extract_info(url, download=False)
                subs = self._subtitles(ydl, data)
        except MediaError:
            raise
        except Exception as e:
            raise self._wrap(url, e) from e
        return MediaInfo(url=url, title=data.get("title") or data.get("fulltitle") or "",
                         caption=data.get("description") or "", subtitles=subs,
                         uploader=data.get("uploader") or data.get("channel"))

    def _subtitles(self, ydl, data: dict) -> Optional[str]:
        # Creator-made subtitles first, then automatic captions.
        for table in (data.get("subtitles") or {}, data.get("automatic_captions") or {}):
            for lang in self.languages:
                tracks = table.get(lang) or next((v for k, v in table.items() if k.startswith(lang + "-")), None)
                vtt = next((t for t in tracks or [] if t.get("ext") == "vtt"), None)
                if vtt:
                    try:
                        return parse_vtt(ydl.urlopen(vtt["url"]).read().decode("utf-8", "replace")) or None
                    except Exception as e:
                        log.info("subtitles not available: %s", e)
        return None

    def download_audio(self, url: str, dest: Path) -> Path:
        # Smallest audio track: plenty for speech, small enough to send inline.
        try:
            with self._ydl(format="worstaudio[ext=m4a]/worstaudio/bestaudio/worst",
                           outtmpl=str(dest / "audio.%(ext)s")) as ydl:
                data = ydl.extract_info(url, download=True)
        except Exception as e:
            raise self._wrap(url, e) from e
        files = [Path(d["filepath"]) for d in data.get("requested_downloads") or [] if d.get("filepath")]
        files = files or sorted(dest.glob("audio.*"))
        if not files:
            raise MediaError(f"no audio could be downloaded from {url}")
        return files[0]


# ------------------------------------------------------------------ transcription
class Transcriber(Protocol):
    name: str

    async def transcribe(self, path: Path) -> str: ...


class DeepgramTranscriber:
    """Deepgram pre-recorded transcription (``POST /v1/listen``)."""

    name = "deepgram"

    def __init__(self, api_key: Optional[str] = None, model: str = "nova-3"):
        self.key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not self.key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")
        self.url = (f"https://api.deepgram.com/v1/listen?model={model}"
                    "&smart_format=true&punctuate=true&paragraphs=true&detect_language=true")

    def _post(self, data: bytes, mime: str) -> dict:
        req = urllib.request.Request(self.url, data=data, method="POST",
                                     headers={"Authorization": f"Token {self.key}", "Content-Type": mime})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())

    async def transcribe(self, path: Path) -> str:
        body = await asyncio.to_thread(self._post, path.read_bytes(), mime_for(path))
        alt = body["results"]["channels"][0]["alternatives"][0]
        return ((alt.get("paragraphs") or {}).get("transcript") or alt.get("transcript") or "").strip()


class _Transcript(BaseModel):
    text: str = Field(description="Everything said, word for word.")


class GeminiTranscriber:
    """Transcription by the LLM itself, for when there is no Deepgram key."""

    name = "gemini"

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def transcribe(self, path: Path) -> str:
        data = path.read_bytes()
        if len(data) > INLINE_LIMIT_BYTES:
            raise MediaError("this audio is too long to transcribe with Gemini directly; set DEEPGRAM_API_KEY")
        out = await self.llm.generate_structured(
            "Transcribe everything said in this recording word for word, in the language spoken. "
            "Do not summarise, translate or correct anything. Use normal punctuation and a new line per sentence.",
            _Transcript, media=[MediaPart(data, mime_for(path))])
        return out.text.strip()


def make_transcriber(llm: Optional[LLMProvider]) -> Optional[Transcriber]:
    if os.environ.get("DEEPGRAM_API_KEY"):
        return DeepgramTranscriber()
    return GeminiTranscriber(llm) if llm is not None else None


async def transcribe_url(url: str, fetcher: MediaFetcher, transcriber: Transcriber) -> str:
    with tempfile.TemporaryDirectory(prefix="cooking-agent-") as d:
        audio = await asyncio.to_thread(fetcher.download_audio, url, Path(d))
        return await transcriber.transcribe(audio)
