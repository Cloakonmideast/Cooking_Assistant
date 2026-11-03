from cooking_agent.segmenter import SentenceSegmenter, clean_for_speech


def feed(seg, text, step=3):
    out = []
    for i in range(0, len(text), step):
        out += seg.push(text[i : i + step])
    return out + seg.flush()


def test_splits_sentences_across_arbitrary_chunking():
    text = "Heat the ghee! Then add cumin? Stir well. Done"
    assert feed(SentenceSegmenter(first_clause_min_chars=999), text) == [
        "Heat the ghee!", "Then add cumin?", "Stir well.", "Done"]


def test_step_numbers_are_not_sentence_breaks():
    out = feed(SentenceSegmenter(first_clause_min_chars=999), "Step 2. Heat the ghee. Next.")
    assert out == ["Step 2. Heat the ghee.", "Next."]


def test_first_segment_can_break_at_a_clause():
    seg = SentenceSegmenter(first_clause_min_chars=20)
    out = seg.push("Sure, I can help with that, let me find a recipe")
    assert out == ["Sure, I can help with that,"]
    # later segments wait for a full sentence
    assert seg.push(" for you, ok") == []


def test_markdown_and_bullets_stripped():
    assert clean_for_speech("**Step 1:** add `salt`") == "Step 1: add salt"
    out = feed(SentenceSegmenter(), "Ingredients:\n- rice\n- ghee\n")
    assert out == ["Ingredients:", "rice", "ghee"]


def test_long_run_on_is_cut():
    seg = SentenceSegmenter(max_chars=40)
    out = seg.push("word " * 30)
    assert out and all(len(s) <= 40 for s in out)
