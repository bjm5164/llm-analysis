"""Exhaustive edge-case tests for tokenization handling.

Covers:
  - Quote ambiguity (tokens that ARE quotes, nested quotes, backticks)
  - Whitespace preservation (leading/trailing spaces, tabs, newlines)
  - Special characters (HTML entities, backslashes, unicode)
  - repr() round-trip fidelity (escaped chars, double-repr)
  - token_id_input parsing edge cases (#ID format, numeric strings)
  - resolve_single_token error paths
  - HTML safety in token display chips
"""

import sys
import html as html_mod
from pathlib import Path

import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app_state import normalize_token_str, resolve_single_token


# ---------------------------------------------------------------------------
# 1. normalize_token_str — quote handling edge cases
# ---------------------------------------------------------------------------

class TestNormalizeQuoteEdgeCases:
    """Tokens that are themselves quote characters, or contain quotes."""

    def test_single_quote_char_bare(self):
        """A bare single-quote character (len 1) — must NOT be stripped."""
        assert normalize_token_str("'") == "'"

    def test_double_quote_char_bare(self):
        """A bare double-quote character (len 1) — must NOT be stripped."""
        assert normalize_token_str('"') == '"'

    def test_single_quote_token_via_double_quotes(self):
        """Token is literally ' — user wraps in double quotes: \"'\"."""
        # This is the correct way to enter a single-quote token
        result = normalize_token_str("\"'\"")
        assert result == "'", f"Expected single quote, got {result!r}"

    def test_double_quote_token_via_single_quotes(self):
        """Token is literally \" — user wraps in single quotes: '\"'."""
        result = normalize_token_str("'\"'")
        assert result == '"', f"Expected double quote, got {result!r}"

    def test_triple_single_quotes_ambiguous(self):
        """Input ''' is ambiguous: could be quoted ' or literal '''.
        Current behavior: strips outer quotes → '."""
        result = normalize_token_str("'''")
        # The outer ' and ' are stripped, leaving middle '
        assert result == "'", (
            "''' is treated as quoted single-quote — "
            "users needing literal ''' must use #ID format"
        )

    def test_token_containing_quotes_in_middle(self):
        """Token like it's — contains an apostrophe."""
        assert normalize_token_str("it's") == "it's"

    def test_token_containing_quotes_wrapped(self):
        """User wraps it's in double quotes."""
        assert normalize_token_str("\"it's\"") == "it's"

    def test_empty_quotes_single(self):
        """Input '' — two single quotes. Strips to empty string."""
        result = normalize_token_str("''")
        assert result == "", f"Expected empty string, got {result!r}"

    def test_empty_quotes_double(self):
        """Input \"\" — two double quotes. Strips to empty string."""
        result = normalize_token_str('""')
        assert result == "", f"Expected empty string, got {result!r}"

    def test_backtick_not_stripped(self):
        """Backticks from markdown display should NOT be stripped."""
        assert normalize_token_str("`hello`") == "`hello`"

    def test_backtick_wrapped_space_token(self):
        """User copies ` 2` from markdown — backticks preserved."""
        result = normalize_token_str("` 2`")
        assert result == "` 2`", "Backticks must not be treated as quotes"

    def test_mixed_quotes_not_stripped(self):
        """Mismatched quote types at start/end — should NOT strip."""
        assert normalize_token_str("'hello\"") == "'hello\""
        assert normalize_token_str("\"hello'") == "\"hello'"

    def test_only_outer_quotes_stripped(self):
        """Inner quotes must survive normalization."""
        assert normalize_token_str("\"'hello'\"") == "'hello'"
        assert normalize_token_str("'\"hello\"'") == '"hello"'


# ---------------------------------------------------------------------------
# 2. normalize_token_str — whitespace edge cases
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:
    """Whitespace is meaningful for tokens — must never be stripped."""

    def test_leading_space(self):
        assert normalize_token_str(" hello") == " hello"

    def test_trailing_space(self):
        assert normalize_token_str("hello ") == "hello "

    def test_both_spaces(self):
        assert normalize_token_str(" hello ") == " hello "

    def test_tab(self):
        assert normalize_token_str("\t") == "\t"

    def test_newline(self):
        assert normalize_token_str("\n") == "\n"

    def test_carriage_return(self):
        assert normalize_token_str("\r") == "\r"

    def test_space_only(self):
        assert normalize_token_str(" ") == " "

    def test_multiple_spaces(self):
        assert normalize_token_str("   ") == "   "

    def test_quoted_leading_space(self):
        """The primary use case: repr(' hello') → \"' hello'\" → ' hello'."""
        assert normalize_token_str("' hello'") == " hello"

    def test_quoted_tab(self):
        """Quoted tab character — but tabs don't survive repr()!"""
        # If user literally types a tab inside quotes:
        assert normalize_token_str("'\t'") == "\t"

    def test_quoted_newline_literal(self):
        """A literal newline inside quotes — unusual but possible."""
        assert normalize_token_str("'\n'") == "\n"

    def test_space_before_quotes_not_stripped(self):
        """Whitespace outside quotes means quotes aren't at [0] and [-1]."""
        result = normalize_token_str("  'hello'  ")
        assert result == "  'hello'  ", "Outer whitespace means no quote-stripping"


# ---------------------------------------------------------------------------
# 3. repr() round-trip — the fundamental tension
# ---------------------------------------------------------------------------

class TestReprRoundTrip:
    """Test that the repr() → normalize pipeline preserves token identity.

    The key issue: repr() escapes special characters, but normalize_token_str
    does NOT unescape them. This means repr() output of tokens with special
    chars CANNOT be round-tripped through normalize_token_str.
    """

    def test_simple_word_roundtrip(self):
        token = "hello"
        assert normalize_token_str(repr(token)) == token

    def test_space_prefix_roundtrip(self):
        token = " France"
        assert normalize_token_str(repr(token)) == token

    def test_digit_roundtrip(self):
        token = "2"
        assert normalize_token_str(repr(token)) == token

    def test_space_digit_roundtrip(self):
        token = " 2"
        assert normalize_token_str(repr(token)) == token

    def test_newline_roundtrip_FAILS(self):
        """KNOWN ISSUE: repr('\\n') = \"'\\\\n'\" → normalize → '\\\\n' ≠ '\\n'."""
        token = "\n"
        normalized = normalize_token_str(repr(token))
        assert normalized != token, (
            "If this passes with ==, normalize_token_str learned to unescape"
        )

    def test_tab_roundtrip_FAILS(self):
        """KNOWN ISSUE: repr('\\t') → normalize gives literal backslash-t."""
        token = "\t"
        normalized = normalize_token_str(repr(token))
        assert normalized != token

    def test_carriage_return_roundtrip_FAILS(self):
        """repr('\\r') → normalize gives literal backslash-r."""
        token = "\r"
        normalized = normalize_token_str(repr(token))
        assert normalized != token

    def test_backslash_roundtrip_FAILS(self):
        """repr('\\\\') = \"'\\\\\\\\\\\\\\\\'\" — backslash is doubled by repr."""
        token = "\\"
        normalized = normalize_token_str(repr(token))
        assert normalized != token, "Backslash is doubled by repr and not un-doubled"

    def test_null_byte_roundtrip_FAILS(self):
        """repr('\\x00') escapes to hex."""
        token = "\x00"
        normalized = normalize_token_str(repr(token))
        assert normalized != token

    def test_unicode_roundtrip(self):
        """Non-ASCII unicode chars are NOT escaped by repr() in Python 3."""
        token = "你好"
        assert normalize_token_str(repr(token)) == token

    def test_emoji_roundtrip(self):
        token = "🔥"
        assert normalize_token_str(repr(token)) == token

    def test_double_repr_strips_one_layer(self):
        """Double-wrapped repr: only outer quotes stripped."""
        token = " 2"
        double = repr(repr(token))  # outer layer wraps in double quotes
        normalized = normalize_token_str(double)
        # After stripping outer double quotes, inner repr remains: ' 2'
        assert normalized == repr(token), f"Got {normalized!r}"
        # This is still wrong — a second normalize would be needed
        assert normalize_token_str(normalized) == token


# ---------------------------------------------------------------------------
# 4. Special character tokens
# ---------------------------------------------------------------------------

class TestSpecialCharTokens:
    """Tokens containing characters that need careful handling."""

    def test_html_angle_brackets(self):
        """Tokens like '<s>' or '</s>' — common special tokens."""
        assert normalize_token_str("<s>") == "<s>"
        assert normalize_token_str("'<s>'") == "<s>"

    def test_ampersand(self):
        assert normalize_token_str("&amp;") == "&amp;"
        assert normalize_token_str("&") == "&"

    def test_hash_in_token_string(self):
        """A # character in a token string — NOT a token ID prefix."""
        # normalize_token_str doesn't handle #, that's token_id_input's job
        assert normalize_token_str("#") == "#"
        assert normalize_token_str("#hello") == "#hello"

    def test_backslash(self):
        assert normalize_token_str("\\") == "\\"

    def test_null_byte(self):
        assert normalize_token_str("\x00") == "\x00"

    def test_unicode_bom(self):
        assert normalize_token_str("\ufeff") == "\ufeff"

    def test_zero_width_space(self):
        """U+200B — invisible but different from empty string."""
        assert normalize_token_str("\u200b") == "\u200b"

    def test_non_breaking_space(self):
        """U+00A0 — looks like a space but isn't."""
        assert normalize_token_str("\u00a0") == "\u00a0"

    def test_ğ_character(self):
        """BPE uses Ġ (U+0120) for space-prefixed tokens in GPT-2."""
        assert normalize_token_str("Ġ") == "Ġ"

    def test_right_to_left_mark(self):
        assert normalize_token_str("\u200f") == "\u200f"


# ---------------------------------------------------------------------------
# 5. HTML safety in token display
# ---------------------------------------------------------------------------

class TestHTMLSafetyInDisplay:
    """Token strings inserted into HTML (e.g., chips in Prompts page)
    must not break rendering. repr() wraps in quotes but does NOT
    escape HTML entities."""

    def _repr_contains_raw_html(self, token: str) -> bool:
        """Check if repr(token) contains unescaped HTML-meaningful chars."""
        r = repr(token)
        # After repr, the string is wrapped in quotes: 'token' or "token"
        # Check if the inner content contains < > & that aren't escaped
        inner = r[1:-1]  # strip outer quotes from repr
        return any(c in inner for c in "<>&")

    def test_angle_bracket_token_unsafe_in_html(self):
        """repr('<script>') = \"'<script>'\" — the < and > are raw HTML."""
        assert self._repr_contains_raw_html("<script>")

    def test_ampersand_token_unsafe_in_html(self):
        assert self._repr_contains_raw_html("&")

    def test_html_escape_preserves_token_identity(self):
        """html.escape() should be applied AFTER repr() for display safety."""
        token = '<script>alert("xss")</script>'
        safe_repr = html_mod.escape(repr(token))
        assert "<" not in safe_repr
        assert ">" not in safe_repr

    def test_common_special_tokens_need_escaping(self):
        """BOS/EOS/pad tokens often have angle brackets."""
        for tok in ["<s>", "</s>", "<pad>", "<unk>", "<|endoftext|>"]:
            assert self._repr_contains_raw_html(tok), (
                f"Token {tok!r} contains raw HTML chars in repr output"
            )


# ---------------------------------------------------------------------------
# 6. Integration tests with model (GPT-2)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    """Load GPT-2 for integration tests."""
    try:
        from transformer_lens import HookedTransformer
        return HookedTransformer.from_pretrained("gpt2", device="cpu")
    except Exception as e:
        pytest.skip(f"Cannot load model: {e}")


class TestResolveEdgeCases:
    """Edge cases for resolve_single_token with a real model."""

    def test_empty_string_raises(self, model):
        """Empty string encodes to zero tokens — must error."""
        with pytest.raises(ValueError, match="is not a single token"):
            resolve_single_token(model, "")

    def test_space_only_is_single_token(self, model):
        """A single space ' ' should be a valid single token in GPT-2."""
        tid = resolve_single_token(model, " ")
        assert isinstance(tid, int)

    def test_newline_is_single_token(self, model):
        """Newline should be a single token."""
        tid = resolve_single_token(model, "\n")
        assert isinstance(tid, int)

    def test_tab_is_single_token(self, model):
        """Tab should be a single token."""
        tid = resolve_single_token(model, "\t")
        assert isinstance(tid, int)

    def test_double_space_may_be_multi_token(self, model):
        """Two spaces might tokenize differently than one."""
        ids = model.tokenizer.encode("  ", add_special_tokens=False)
        # Just document the behavior — don't assert single/multi
        assert len(ids) >= 1

    def test_quote_char_is_single_token(self, model):
        """A single quote character should be a token in GPT-2."""
        # This tests that we can handle the quote char itself
        tid_single = resolve_single_token(model, "'")
        assert isinstance(tid_single, int)

        tid_double = resolve_single_token(model, '"')
        assert isinstance(tid_double, int)

    def test_quote_tokens_are_different(self, model):
        """Single and double quotes should be different tokens."""
        tid_single = resolve_single_token(model, "'")
        tid_double = resolve_single_token(model, '"')
        assert tid_single != tid_double

    def test_normalize_then_resolve_quote_token(self, model):
        """User wants token ' (single quote) — wraps in double quotes: \"'\"."""
        user_input = "\"'\""
        normalized = normalize_token_str(user_input)
        assert normalized == "'"
        tid = resolve_single_token(model, normalized)
        assert isinstance(tid, int)

    def test_normalize_then_resolve_double_quote_token(self, model):
        """User wants token \" (double quote) — wraps in single quotes: '\"'."""
        user_input = "'\"'"
        normalized = normalize_token_str(user_input)
        assert normalized == '"'
        tid = resolve_single_token(model, normalized)
        assert isinstance(tid, int)

    def test_space_vs_no_space_distinction(self, model):
        """Critical: ' the' and 'the' MUST resolve to different IDs."""
        # 'the' without space is typically BOS position, ' the' is normal
        tid_bare = resolve_single_token(model, "the")
        tid_space = resolve_single_token(model, " the")
        assert tid_bare != tid_space, (
            "Leading space distinction lost — this breaks all interpretability claims"
        )

    def test_decode_roundtrip_preserves_identity(self, model):
        """encode → decode → encode must return the same ID."""
        test_tokens = [" the", "hello", "\n", " 2", "'", '"']
        for tok_str in test_tokens:
            ids = model.tokenizer.encode(tok_str, add_special_tokens=False)
            if len(ids) != 1:
                continue  # skip multi-token strings
            decoded = model.tokenizer.decode(ids[0])
            re_encoded = model.tokenizer.encode(decoded, add_special_tokens=False)
            assert re_encoded == ids, (
                f"Round-trip failed for {tok_str!r}: "
                f"id={ids[0]} → decoded={decoded!r} → re-encoded={re_encoded}"
            )

    def test_repr_copied_from_explorer_roundtrip(self, model):
        """Simulate copying repr output from the token explorer and pasting it back."""
        # The explorer shows repr(decoded_token) for each position
        test_strs = [" the", "hello", " 2", "'"]
        for original in test_strs:
            ids = model.tokenizer.encode(original, add_special_tokens=False)
            if len(ids) != 1:
                continue
            # Explorer displays repr(original)
            displayed = repr(original)
            # User copies this and pastes into token_id_input
            normalized = normalize_token_str(displayed)
            # Should resolve to the same ID
            resolved_id = resolve_single_token(model, normalized)
            assert resolved_id == ids[0], (
                f"Explorer display round-trip failed: {original!r} → "
                f"displayed={displayed!r} → normalized={normalized!r} → "
                f"id={resolved_id} (expected {ids[0]})"
            )


# ---------------------------------------------------------------------------
# 7. Qwen tokenizer edge cases (no full model needed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Load the Qwen3-4B tokenizer."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Cannot load Qwen tokenizer: {e}")


class TestQwenEdgeCases:
    """Qwen-specific tokenization pitfalls."""

    def test_space_is_token_220(self, qwen_tokenizer):
        """Space is a specific token in Qwen — verify ID is stable."""
        ids = qwen_tokenizer.encode(" ", add_special_tokens=False)
        assert len(ids) == 1
        assert ids[0] == 220, f"Space token ID changed: expected 220, got {ids[0]}"

    def test_individual_decode_matches_prompt_context(self, qwen_tokenizer):
        """Individual decode of each token should match what the tokenizer
        actually produced — this is the foundation of the token explorer."""
        prompt = "What is 1 + 1? The answer is 2."
        ids = qwen_tokenizer.encode(prompt, add_special_tokens=False)
        # Decode individually
        individual = [qwen_tokenizer.decode(tid) for tid in ids]
        # Re-encode each individually decoded token
        for i, (tid, decoded) in enumerate(zip(ids, individual)):
            re_ids = qwen_tokenizer.encode(decoded, add_special_tokens=False)
            # The re-encoded might be multiple tokens if the tokenizer
            # merged context-dependently. This is a known tokenizer behavior.
            if len(re_ids) == 1:
                assert re_ids[0] == tid, (
                    f"Position {i}: decode({tid})={decoded!r} re-encodes to "
                    f"{re_ids} instead of [{tid}]"
                )

    def test_quote_tokens_in_qwen(self, qwen_tokenizer):
        """Verify that quote characters are single tokens."""
        for char in ["'", '"', "`"]:
            ids = qwen_tokenizer.encode(char, add_special_tokens=False)
            assert len(ids) == 1, f"Quote {char!r} is not a single token: {ids}"

    def test_special_chars_tokenization(self, qwen_tokenizer):
        """Document how special characters tokenize in Qwen."""
        chars = {
            "\n": "newline",
            "\t": "tab",
            " ": "space",
            "\\": "backslash",
            "<": "angle bracket",
            ">": "angle bracket",
            "&": "ampersand",
        }
        for char, desc in chars.items():
            ids = qwen_tokenizer.encode(char, add_special_tokens=False)
            decoded = qwen_tokenizer.decode(ids[0]) if ids else ""
            # Just verify encoding doesn't crash and produces at least 1 token
            assert len(ids) >= 1, f"{desc} ({char!r}) produced 0 tokens"

    def test_user_input_with_curly_quotes(self, qwen_tokenizer):
        """Users might paste text with Unicode curly quotes instead of ASCII."""
        # Curly quotes are different characters from straight quotes
        for q in ["\u2018", "\u2019", "\u201c", "\u201d"]:  # ' ' " "
            ids = qwen_tokenizer.encode(q, add_special_tokens=False)
            assert len(ids) >= 1, f"Curly quote {q!r} produced 0 tokens"

    def test_normalize_does_not_strip_curly_quotes(self):
        """Curly quotes should NOT be treated as wrapping quotes."""
        assert normalize_token_str("\u2018hello\u2019") == "\u2018hello\u2019"
        assert normalize_token_str("\u201chello\u201d") == "\u201chello\u201d"

    def test_prompt_with_quotes_tokenizes_correctly(self, qwen_tokenizer):
        """A prompt containing quotes should tokenize without issues."""
        prompt = 'He said "hello" and she said \'goodbye\''
        ids = qwen_tokenizer.encode(prompt, add_special_tokens=False)
        individual = [qwen_tokenizer.decode(tid) for tid in ids]
        # Concatenation should reconstruct the original
        reconstructed = "".join(individual)
        assert reconstructed == prompt, (
            f"Reconstruction failed:\n  original:      {prompt!r}\n"
            f"  reconstructed: {reconstructed!r}\n"
            f"  tokens: {list(zip(ids, individual))}"
        )

    def test_answer_token_with_surrounding_context(self, qwen_tokenizer):
        """Verify that extracting an answer token by individual decode
        gives the correct string for re-encoding."""
        prompt = "1+1="
        ids = qwen_tokenizer.encode(prompt, add_special_tokens=False)
        # Each token decoded individually — this is what the explorer shows
        individual = [qwen_tokenizer.decode(tid) for tid in ids]
        # User would see these in the explorer and copy one
        for tid, decoded in zip(ids, individual):
            repr_display = repr(decoded)
            normalized = normalize_token_str(repr_display)
            re_ids = qwen_tokenizer.encode(normalized, add_special_tokens=False)
            if len(re_ids) == 1:
                assert re_ids[0] == tid, (
                    f"Explorer → copy → paste failed: "
                    f"id={tid}, decoded={decoded!r}, repr={repr_display}, "
                    f"normalized={normalized!r}, re_encoded={re_ids}"
                )


# ---------------------------------------------------------------------------
# 8. token_id_input parsing logic (unit tests without Streamlit)
# ---------------------------------------------------------------------------

class TestTokenIdParsing:
    """Test the parsing logic inside token_id_input, extracted for unit testing.

    Since token_id_input relies on st.text_input, we test the parsing
    logic it applies to the raw string.
    """

    def test_hash_prefix_extraction(self):
        """#123 → extract 123 as integer."""
        raw = "#220"
        assert raw.startswith("#")
        id_part = raw[1:].strip()
        assert id_part == "220"
        assert id_part.isdigit()
        assert int(id_part) == 220

    def test_hash_with_spaces(self):
        """# 220 → strip → 220."""
        raw = "# 220"
        id_part = raw[1:].strip()
        assert id_part == "220"

    def test_hash_only(self):
        """Just '#' — no digits."""
        raw = "#"
        id_part = raw[1:].strip()
        assert id_part == ""
        assert not id_part.isdigit()

    def test_hash_with_text(self):
        """#abc — not a valid ID."""
        raw = "#abc"
        id_part = raw[1:].strip()
        assert not id_part.isdigit()

    def test_hash_negative(self):
        """#-1 — negative, not valid."""
        raw = "#-1"
        id_part = raw[1:].strip()
        assert not id_part.isdigit()

    def test_hash_float(self):
        """#1.5 — float, not valid."""
        raw = "#1.5"
        id_part = raw[1:].strip()
        assert not id_part.isdigit()

    def test_hash_zero(self):
        """#0 — valid token ID."""
        raw = "#0"
        id_part = raw[1:].strip()
        assert id_part.isdigit()
        assert int(id_part) == 0

    def test_numeric_string_without_hash(self):
        """'220' without # — treated as token string, not ID."""
        raw = "220"
        assert not raw.startswith("#")
        # This goes to normalize_token_str → encode path
        normalized = normalize_token_str(raw)
        assert normalized == "220"

    def test_quoted_hash_id(self):
        """'#220' with quotes — startswith('#') is False."""
        raw = "'#220'"
        assert not raw.startswith("#")
        # Goes to token string path
        normalized = normalize_token_str(raw)
        assert normalized == "#220"

    def test_isdigit_hint_for_numeric_tokens(self):
        """When a numeric string isn't a single token, the code shows a hint."""
        s = "220"
        normalized = normalize_token_str(s)
        assert normalized.isdigit(), "numeric hint should trigger"

    def test_isdigit_hint_not_for_text(self):
        """Non-numeric tokens should not trigger the 'did you mean #ID' hint."""
        s = "hello"
        normalized = normalize_token_str(s)
        assert not normalized.isdigit()


# ---------------------------------------------------------------------------
# 9. answer_residual_direction validation gap
# ---------------------------------------------------------------------------

class TestAnswerResidualDirection:
    """Test that answer_residual_direction handles multi-token strings."""

    def test_multi_token_answer_raises(self, model):
        """Multi-token string must raise ValueError, not silently produce
        a wrong-shaped direction vector."""
        from attribution import answer_residual_direction

        with pytest.raises(ValueError, match="is not a single token"):
            answer_residual_direction(model, "hello world")

    def test_single_token_answer_correct_shape(self, model):
        """Single-token string should return (d_model,) vector."""
        from attribution import answer_residual_direction

        direction = answer_residual_direction(model, " the")
        assert direction.ndim == 1, f"Expected 1D, got shape {direction.shape}"
        assert direction.shape[0] == model.cfg.d_model


# ---------------------------------------------------------------------------
# 10. Tokenize function — individual decode correctness
# ---------------------------------------------------------------------------

class TestTokenizeFunction:
    """Test that model.py:tokenize() decodes tokens individually."""

    def test_individual_decode_vs_batch(self, model):
        """Individual decode should differ from batch decode for some prompts.
        This verifies we're using the right decoding strategy."""
        from model import tokenize

        prompt = "Hello, world!"
        tokens, str_tokens = tokenize(model, prompt, prepend_bos=True)

        # Individual decode (what tokenize does)
        individual = [model.tokenizer.decode(t.item()) for t in tokens[0]]

        # Batch decode (what we must NOT do)
        batch = model.tokenizer.decode(tokens[0].tolist())

        # Individual tokens concatenated should reconstruct the prompt
        # (possibly with BOS prefix)
        concat = "".join(str_tokens[1:])  # skip BOS
        # Note: this might not exactly match prompt due to tokenizer quirks
        # but the important thing is individual == our str_tokens
        assert individual == str_tokens

    def test_tokens_and_str_tokens_same_length(self, model):
        """token IDs and string tokens must have the same count."""
        from model import tokenize

        prompt = "The answer is 42."
        tokens, str_tokens = tokenize(model, prompt, prepend_bos=True)
        assert len(str_tokens) == tokens.shape[1]

    def test_bos_token_present_when_prepended(self, model):
        """First token should be BOS when prepend_bos=True."""
        from model import tokenize

        tokens_bos, _ = tokenize(model, "hello", prepend_bos=True)
        tokens_no_bos, _ = tokenize(model, "hello", prepend_bos=False)

        assert tokens_bos.shape[1] == tokens_no_bos.shape[1] + 1
