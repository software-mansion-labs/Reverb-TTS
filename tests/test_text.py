import pytest

from reverb_tts.text import estimate_frames, prepare_text, split_text


def test_prepare_text_reads_numbers_out_in_the_training_character_forms() -> None:
    assert prepare_text("I paid $5 – “now”…", "en") == 'I paid five dollars - "now"...'
    assert prepare_text("Kupiłem 3 bilety.", "pl") == "Kupiłem trzy bilety."


def test_prepare_text_drops_brackets() -> None:
    assert prepare_text("Prix 3 € (environ).", "fr") == "Prix trois euros environ."


def test_unsupported_language_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported language"):
        prepare_text("Hello", "nl")


def test_estimate_frames_follows_the_reference_rate() -> None:
    reference = "A reference transcript of moderate length."
    long_text = "word " * 40
    assert estimate_frames(long_text, reference, 100) == pytest.approx(2 * estimate_frames(long_text, reference, 50), abs=1)
    assert estimate_frames(long_text + long_text, reference, 100) == pytest.approx(2 * estimate_frames(long_text, reference, 100), abs=2)
    # Short texts get relatively more time than a linear estimate gives them.
    assert estimate_frames("Hi.", reference, 100) > 100 * 3 / len(reference)


def test_split_text_cuts_only_long_text_at_sentence_ends() -> None:
    reference = "Ten words of reference text spoken at a steady pace here."
    sentences = [f"This is sentence number {i} of a long text." for i in range(30)]
    text = " ".join(sentences)
    assert split_text("One short sentence.", reference, 250, longest=750, piece=375) == ["One short sentence."]
    pieces = split_text(text, reference, 250, longest=750, piece=375)
    assert len(pieces) > 1 and " ".join(pieces) == text
    assert all(estimate_frames(p, reference, 250) <= 375 for p in pieces)
