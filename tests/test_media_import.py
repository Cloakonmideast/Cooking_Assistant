"""Importing recipes from social media, videos, photos and pages, word for word. No network."""

from pathlib import Path

import pytest

from cooking_agent.media_import import (
    DeepgramTranscriber,
    MediaInfo,
    is_media_url,
    parse_vtt,
)
from cooking_agent.recipe_import import (
    ExtractedRecipe,
    RecipeImportError,
    _tokens,
    import_recipe,
    is_verbatim,
)

CAPTION = """Jeera rice in 15 mins 🔥🍚 Save this!
Serves 2
Ingredients:
• 1 cup basmati rice, soaked
• 1½ tbsp ghee
• 1 tsp cumin seeds
Method:
1. Heat ghee and crackle the cumin seeds.
2. Add the rice and 2 cups of water, cover and cook for 12 minutes.
#jeerarice #easyrecipes link in bio"""

GOOD = ExtractedRecipe(found=True, title="Jeera rice", servings=2,
                       ingredients=["1 cup basmati rice, soaked", "1½ tbsp ghee", "1 tsp cumin seeds"],
                       steps=["Heat ghee and crackle the cumin seeds.",
                              "Add the rice and 2 cups of water, cover and cook for 12 minutes."])
REWORDED = GOOD.model_copy(update={"steps": ["Heat the ghee and fry cumin until it pops.", GOOD.steps[1]]})
NOTHING = ExtractedRecipe(found=False)


class FakeLLM:
    name = "fake"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts, self.media = [], []

    async def generate_structured(self, prompt, schema, media=None):
        self.prompts.append(prompt)
        self.media.append(media)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


class FakeFetcher:
    def __init__(self, caption="", subtitles=None):
        self.caption, self.subtitles = caption, subtitles
        self.downloaded = False

    def info(self, url):
        return MediaInfo(url=url, title="Jeera Rice Reel", caption=self.caption, subtitles=self.subtitles)

    def download_audio(self, url, dest: Path):
        self.downloaded = True
        f = dest / "audio.m4a"
        f.write_bytes(b"fake audio")
        return f


class FakeTranscriber:
    name = "fake"

    def __init__(self, text):
        self.text, self.calls = text, []

    async def transcribe(self, path):
        self.calls.append(path)
        return self.text


# ------------------------------------------------------------------ helpers
def test_is_media_url():
    for url in ["https://www.youtube.com/watch?v=abc", "https://youtu.be/abc", "https://youtube.com/shorts/x",
                "https://www.instagram.com/reel/Cx/", "https://www.tiktok.com/@chef/video/1", "https://fb.watch/x/"]:
        assert is_media_url(url), url
    assert not is_media_url("https://www.bbcgoodfood.com/recipes/pancakes")


def test_parse_vtt_drops_timings_tags_and_scroll_repeats():
    vtt = ("WEBVTT\nKind: captions\nLanguage: en\n\n00:00:00.000 --> 00:00:02.000\nfirst take <c>one</c> cup rice\n\n"
           "00:00:02.000 --> 00:00:04.000\nfirst take one cup rice\n\n00:00:04.000 --> 00:00:06.000\nthen wash it")
    assert parse_vtt(vtt) == "first take one cup rice\nthen wash it"


@pytest.mark.parametrize("line,ok", [
    ("Heat ghee and crackle the cumin seeds.", True),
    ("1½ tbsp ghee", True),
    ("1 1/2 tbsp ghee", True),               # same amount, other spelling of the fraction
    ("Add the rice and 2 cups of water", True),
    ("Heat the ghee and fry cumin until it pops.", False),  # reworded
    ("1.5 tbsp ghee", False),               # changed number format = not word for word
])
def test_is_verbatim(line, ok):
    assert is_verbatim(line, _tokens(CAPTION)) is ok


# ------------------------------------------------------------------ social media / video links
async def test_caption_with_full_recipe_is_used_as_is():
    llm, fetcher, tr = FakeLLM(GOOD), FakeFetcher(CAPTION), FakeTranscriber("")
    res = await import_recipe("https://www.instagram.com/reel/abc/", llm=llm, transcriber=tr, fetcher=fetcher)
    assert res.how == "the post's caption/description"
    assert res.warnings == []
    assert [i.describe() for i in res.recipe.ingredients] == GOOD.ingredients  # word for word
    assert [s.instruction for s in res.recipe.steps] == GOOD.steps
    assert res.recipe.steps[1].timer_minutes == 12
    assert not fetcher.downloaded and tr.calls == []
    assert "copy it out EXACTLY" in llm.prompts[0] and CAPTION in llm.prompts[0]


async def test_reworded_lines_are_sent_back_and_fixed():
    llm = FakeLLM(REWORDED, GOOD)
    res = await import_recipe("https://youtu.be/x", llm=llm, transcriber=None, fetcher=FakeFetcher(CAPTION))
    assert len(llm.prompts) == 2 and "Heat the ghee and fry cumin until it pops." in llm.prompts[1]
    assert res.warnings == [] and res.recipe.steps[0].instruction == GOOD.steps[0]


async def test_lines_still_reworded_are_reported():
    res = await import_recipe("https://youtu.be/x", llm=FakeLLM(REWORDED), fetcher=FakeFetcher(CAPTION))
    assert any("not word for word" in w and "fry cumin" in w for w in res.warnings)


async def test_spoken_recipe_uses_subtitles_then_audio():
    spoken = ("so first take 1 cup basmati rice and soak it. um then heat 1 tbsp ghee. "
              "add the rice and cook for 12 minutes")
    ex = ExtractedRecipe(found=True, title="Jeera rice", servings=None,
                         ingredients=["1 cup basmati rice", "1 tbsp ghee"],
                         steps=["take 1 cup basmati rice and soak it.", "then heat 1 tbsp ghee.",
                                "add the rice and cook for 12 minutes"])
    # caption has no recipe -> subtitles
    llm = FakeLLM(NOTHING, ex)
    res = await import_recipe("https://youtu.be/x", llm=llm, fetcher=FakeFetcher("Best rice ever!! 😍", spoken))
    assert res.how == "the caption and the video's subtitles"
    assert "WHAT IS SAID IN THE VIDEO" in llm.prompts[1] and spoken in llm.prompts[1]
    assert any("assumed 4" in w for w in res.warnings)  # servings not stated
    # no caption, no subtitles -> download the audio and transcribe it
    fetcher, tr = FakeFetcher("", None), FakeTranscriber(spoken)
    res = await import_recipe("https://www.tiktok.com/@a/video/1", llm=FakeLLM(ex), transcriber=tr, fetcher=fetcher)
    assert fetcher.downloaded and len(tr.calls) == 1
    assert not tr.calls[0].exists()  # temporary audio deleted afterwards
    assert res.how == "the spoken audio (transcribed)"


async def test_no_recipe_anywhere_is_a_clear_error():
    with pytest.raises(RecipeImportError, match="no recipe found"):
        await import_recipe("https://youtu.be/x", llm=FakeLLM(NOTHING), fetcher=FakeFetcher("vlog", "hello"))
    with pytest.raises(RecipeImportError, match="GEMINI_API_KEY"):
        await import_recipe("https://youtu.be/x", llm=None, fetcher=FakeFetcher(CAPTION))


# ------------------------------------------------------------------ files and pages
async def test_photo_and_pdf_go_to_the_llm_as_media(tmp_path):
    img = tmp_path / "card.jpg"
    img.write_bytes(b"\xff\xd8 fake jpeg")
    llm = FakeLLM(GOOD)
    res = await import_recipe(str(img), llm=llm)
    assert llm.media[0][0].mime_type == "image/jpeg" and llm.media[0][0].data == img.read_bytes()
    assert any("couldn't be checked" in w for w in res.warnings)
    pdf = tmp_path / "recipe.pdf"
    pdf.write_bytes(b"%PDF fake")
    await import_recipe(str(pdf), llm=llm)
    assert llm.media[-1][0].mime_type == "application/pdf"


async def test_audio_or_video_file_is_transcribed(tmp_path):
    clip = tmp_path / "nani.mp4"
    clip.write_bytes(b"fake")
    tr = FakeTranscriber(CAPTION)
    res = await import_recipe(str(clip), llm=FakeLLM(GOOD), transcriber=tr)
    assert tr.calls == [clip] and "transcribed" in res.how


async def test_pasted_caption_in_a_text_file(tmp_path):
    f = tmp_path / "reel.txt"
    f.write_text(CAPTION)
    res = await import_recipe(str(f), llm=FakeLLM(GOOD))
    assert res.how == "the text file" and res.recipe.title == "Jeera rice"


async def test_page_without_recipe_data_falls_back_to_its_text(tmp_path):
    page = tmp_path / "blog.html"
    page.write_text("<html><script>var x=1</script><body><h1>My jeera rice</h1><p>"
                    + CAPTION.replace("\n", "<br>") + "</p></body></html>")
    llm = FakeLLM(GOOD)
    res = await import_recipe(str(page), llm=llm)
    assert res.how == "the text of the saved web page"
    assert "var x" not in llm.prompts[0] and "crackle the cumin" in llm.prompts[0]


# ------------------------------------------------------------------ transcription / gemini media
async def test_deepgram_transcriber_prefers_paragraphs(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "k")
    f = tmp_path / "a.m4a"
    f.write_bytes(b"x")
    t = DeepgramTranscriber()
    seen = {}

    def fake_post(data, mime):
        seen["mime"] = mime
        return {"results": {"channels": [{"alternatives": [
            {"transcript": "flat", "paragraphs": {"transcript": "\nPara one.\n\nPara two."}}]}]}}

    monkeypatch.setattr(t, "_post", fake_post)
    assert await t.transcribe(f) == "Para one.\n\nPara two."
    assert seen["mime"] == "audio/mp4"
    assert "detect_language=true" in t.url


async def test_gemini_sends_media_before_the_prompt(monkeypatch):
    pytest.importorskip("google.genai")
    import types as pytypes

    from cooking_agent.providers.base import MediaPart
    from cooking_agent.providers.gemini import GeminiLLM

    llm = GeminiLLM(api_key="test", model="m")
    calls = []

    async def generate_content(**kw):
        calls.append(kw)
        return pytypes.SimpleNamespace(text=GOOD.model_dump_json())

    llm.client = pytypes.SimpleNamespace(aio=pytypes.SimpleNamespace(
        models=pytypes.SimpleNamespace(generate_content=generate_content)))
    out = await llm.generate_structured("find it", ExtractedRecipe, media=[MediaPart(b"img", "image/png")])
    assert out == GOOD
    contents = calls[0]["contents"]
    assert contents[0].inline_data.mime_type == "image/png" and contents[-1] == "find it"
