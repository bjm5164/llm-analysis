"""Tests for token string normalization and the to_single_token pipeline.

Reproduces the "input string is not a single token" error that occurs when
token strings from the UI (prompt explorer, repr() output) are passed through
normalize_token_str and then into model.to_single_token().
"""

import sys
from pathlib import Path

import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app_state import normalize_token_str, resolve_single_token


# ---------------------------------------------------------------------------
# Pure normalize_token_str tests (no model needed)
# ---------------------------------------------------------------------------

class TestNormalizeTokenStr:
    """Test the normalize_token_str utility in isolation."""

    def test_plain_string(self):
        assert normalize_token_str("hello") == "hello"

    def test_single_quoted(self):
        """Strings copied from repr() output: ' 2'"""
        assert normalize_token_str("' 2'") == " 2"

    def test_double_quoted(self):
        assert normalize_token_str('" 2"') == " 2"

    def test_whitespace_preserved(self):
        """Whitespace is NOT stripped — it's meaningful for tokens."""
        assert normalize_token_str("  hello  ") == "  hello  "

    def test_whitespace_outside_quotes(self):
        """Outer whitespace preserved, quotes stripped."""
        assert normalize_token_str("  ' 2'  ") == "  ' 2'  "  # no match: outer chars aren't quotes

    def test_no_quotes(self):
        assert normalize_token_str("2") == "2"

    def test_leading_space_preserved(self):
        """Leading space is meaningful — ' 2' and '2' are different tokens."""
        assert normalize_token_str(" 2") == " 2"

    def test_leading_space_preserved_in_quotes(self):
        """The only way to pass a leading-space token is inside quotes."""
        assert normalize_token_str("' 2'") == " 2"

    def test_empty_string(self):
        assert normalize_token_str("") == ""

    def test_single_char(self):
        assert normalize_token_str("x") == "x"

    def test_mismatched_quotes(self):
        """Mismatched quotes should NOT be stripped."""
        assert normalize_token_str("'hello\"") == "'hello\""

    def test_nested_quotes(self):
        """repr(repr()) — double-wrapped."""
        result = normalize_token_str("\"' 2'\"")
        # Outer double quotes stripped → ' 2'
        # But the inner quotes remain — this is a problem!
        assert result == "' 2'"

    def test_repr_output_format(self):
        """What repr() actually produces for common tokens."""
        # repr(' 2') produces "' 2'" in Python
        token = ' 2'
        repr_str = repr(token)  # "' 2'"
        normalized = normalize_token_str(repr_str)
        assert normalized == ' 2', f"repr({token!r}) = {repr_str!r} → normalized = {normalized!r}"

    def test_repr_of_newline(self):
        """repr of a newline token."""
        token = '\n'
        repr_str = repr(token)  # "'\\n'"
        normalized = normalize_token_str(repr_str)
        # After stripping quotes: \\n (the literal backslash-n, not a newline)
        # This will NOT match the original token
        assert normalized != token, "normalize_token_str can't handle escaped repr output"

    def test_repr_of_tab(self):
        token = '\t'
        repr_str = repr(token)  # "'\\t'"
        normalized = normalize_token_str(repr_str)
        assert normalized != token, "normalize_token_str can't handle escaped repr output"


# ---------------------------------------------------------------------------
# Integration tests with model.to_single_token (require GPU / model)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    """Load a small model for token tests. Skip if not available."""
    try:
        from transformer_lens import HookedTransformer
        m = HookedTransformer.from_pretrained(
            "gpt2",  # small, widely available
            device="cpu",
        )
        return m
    except Exception as e:
        pytest.skip(f"Cannot load model: {e}")


class TestTokenRoundTrip:
    """Test the full pipeline: UI string → normalize → to_single_token."""

    def test_plain_token(self, model):
        """A simple token that doesn't need a leading space."""
        tok_id = model.to_single_token("the")
        assert isinstance(tok_id, int)

    def test_space_prefixed_token(self, model):
        """Most tokens in GPT-2 have a leading space (Ġ in BPE)."""
        tok_id = model.to_single_token(" the")
        assert isinstance(tok_id, int)

    def test_digit_without_space(self, model):
        """'2' without leading space — may or may not be a single token."""
        try:
            tok_id = model.to_single_token("2")
            assert isinstance(tok_id, int)
        except Exception:
            pytest.skip("'2' is not a single token in this model")

    def test_digit_with_space(self, model):
        """' 2' with leading space — the common BPE form."""
        tok_id = model.to_single_token(" 2")
        assert isinstance(tok_id, int)

    def test_normalize_then_single_token_quoted(self, model):
        """Simulates: user copies ' 2' from repr output, types it with quotes."""
        user_input = "' 2'"
        normalized = normalize_token_str(user_input)
        assert normalized == " 2"
        tok_id = model.to_single_token(normalized)
        assert isinstance(tok_id, int)

    def test_normalize_then_single_token_bare(self, model):
        """User types '2' without quotes — strip() removes any leading space."""
        user_input = "2"
        normalized = normalize_token_str(user_input)
        assert normalized == "2"
        # This might fail if '2' needs a leading space to be a single token
        try:
            tok_id = model.to_single_token(normalized)
            assert isinstance(tok_id, int)
        except Exception as e:
            # This IS the bug: user sees ' 2' in explorer, types 2,
            # normalize strips the space, to_single_token fails
            pytest.xfail(f"Expected failure: bare '2' is not a single token: {e}")

    def test_normalize_preserves_leading_space(self, model):
        """Leading space must be preserved — ' 2' and '2' are different tokens."""
        user_input = " 2"
        normalized = normalize_token_str(user_input)
        assert normalized == " 2", "leading space must be preserved"

    def test_leading_space_via_quotes_also_works(self, model):
        """Quoting is another way to enter a leading-space token."""
        quoted_input = "' 2'"
        normalized = normalize_token_str(quoted_input)
        assert normalized == " 2"

    def test_double_wrapped_repr(self, model):
        """If the token goes through repr() twice somehow."""
        token = " 2"
        double_repr = repr(repr(token))  # "\"' 2'\""
        normalized = normalize_token_str(double_repr)
        # Only outer quotes stripped: ' 2' remains as a string with quotes
        assert normalized == "' 2'"
        # This will fail to_single_token because of the inner quotes
        with pytest.raises(Exception):
            model.to_single_token(normalized)

    def test_format_from_prompt_explorer(self, model):
        """The prompt explorer shows tokens via repr(). When user copies,
        they get the repr string including quotes. Test common patterns."""
        # Pattern 1: repr of a space-prefixed token
        explorer_display = repr(" 2")  # "' 2'"
        normalized = normalize_token_str(explorer_display)
        tok_id = model.to_single_token(normalized)
        assert isinstance(tok_id, int)

        # Pattern 2: repr of a token without space
        explorer_display = repr("the")  # "'the'"
        normalized = normalize_token_str(explorer_display)
        tok_id = model.to_single_token(normalized)
        assert isinstance(tok_id, int)


# ---------------------------------------------------------------------------
# Qwen-specific tokenizer tests (no full model load needed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Load the Qwen3-4B tokenizer. Skip if not available."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Cannot load Qwen tokenizer: {e}")


class TestQwenTokenization:
    """Qwen3-4B specific tests — this is the model used in production.

    KEY FINDING: In Qwen3-4B, ' 2' (space + digit) is TWO tokens [220, 17],
    not one. GPT-2 merges it into a single token, but Qwen does not.
    This means normalize_token_str("' 2'") → " 2" → to_single_token FAILS.
    """

    def test_digit_is_single_token(self, qwen_tokenizer):
        """'2' (no space) IS a single token in Qwen."""
        ids = qwen_tokenizer.encode("2", add_special_tokens=False)
        assert len(ids) == 1, f"'2' should be 1 token, got {ids}"

    def test_space_digit_is_two_tokens(self, qwen_tokenizer):
        """' 2' (space + digit) is TWO tokens in Qwen — this is the root cause."""
        ids = qwen_tokenizer.encode(" 2", add_special_tokens=False)
        assert len(ids) == 2, f"' 2' should be 2 tokens in Qwen, got {ids}"

    def test_space_is_own_token(self, qwen_tokenizer):
        """The space is token 220 in Qwen."""
        ids = qwen_tokenizer.encode(" ", add_special_tokens=False)
        assert len(ids) == 1, f"' ' should be 1 token, got {ids}"

    def test_normalize_quoted_space_digit_fails_single_token(self, qwen_tokenizer):
        """THE BUG: user copies repr(' 2') = \"' 2'\" from explorer.
        normalize_token_str strips quotes → ' 2'.
        But ' 2' is 2 tokens in Qwen, so to_single_token will fail."""
        user_input = "' 2'"  # copied from explorer repr output
        normalized = normalize_token_str(user_input)
        assert normalized == " 2"
        ids = qwen_tokenizer.encode(normalized, add_special_tokens=False)
        assert len(ids) == 2, (
            f"This is the bug: normalized=' 2' tokenizes to {ids} "
            f"(2 tokens), but to_single_token expects exactly 1"
        )

    def test_bare_digit_works(self, qwen_tokenizer):
        """Without the space, '2' works fine as a single token."""
        user_input = "2"
        normalized = normalize_token_str(user_input)
        assert normalized == "2"
        ids = qwen_tokenizer.encode(normalized, add_special_tokens=False)
        assert len(ids) == 1

    def test_common_answer_tokens(self, qwen_tokenizer):
        """Test various tokens users might enter as answer tokens."""
        single_tokens = []
        multi_tokens = []
        for s in ["2", " 2", "Yes", " Yes", "true", " true", "A", " A"]:
            ids = qwen_tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                single_tokens.append(s)
            else:
                multi_tokens.append((s, ids))

        # Document which are single vs multi for Qwen
        print(f"\nSingle tokens: {single_tokens}")
        print(f"Multi tokens: {multi_tokens}")

    def test_explorer_shows_individual_decode(self, qwen_tokenizer):
        """The explorer decodes each token individually.
        For a prompt containing '2', the explorer shows repr('2') = \"'2'\".
        NOT repr(' 2'), because the space is a separate token.
        So the user should copy '2', not ' 2'."""
        # Simulate what tokenize() does: decode each token individually
        prompt = "The answer is 2"
        ids = qwen_tokenizer.encode(prompt, add_special_tokens=False)
        str_tokens = [qwen_tokenizer.decode(tid) for tid in ids]

        # Find the '2' token
        two_tokens = [t for t in str_tokens if t.strip() == "2"]
        assert len(two_tokens) >= 1, f"Expected to find '2' in {str_tokens}"

        # The explorer displays repr() of each decoded token
        # The '2' token decodes to '2' (no space), so repr is "'2'"
        for t in two_tokens:
            repr_display = repr(t)
            normalized = normalize_token_str(repr_display)
            result_ids = qwen_tokenizer.encode(normalized, add_special_tokens=False)
            assert len(result_ids) == 1, (
                f"Token {t!r} displayed as {repr_display}, "
                f"normalized to {normalized!r}, "
                f"but encodes to {result_ids} (not single token!)"
            )


# ---------------------------------------------------------------------------
# Tests for resolve_single_token (the fix)
# ---------------------------------------------------------------------------

class TestResolveSingleToken:
    """Test the token resolver — no silent fallbacks, preserves spaces exactly."""

    def test_plain_token(self, model):
        tid = resolve_single_token(model, "the")
        assert isinstance(tid, int)

    def test_space_prefixed_token(self, model):
        """GPT-2 merges ' 2' into one token."""
        tid = resolve_single_token(model, " 2")
        assert isinstance(tid, int)

    def test_bare_digit(self, model):
        """'2' is a single token in GPT-2."""
        tid = resolve_single_token(model, "2")
        assert isinstance(tid, int)

    def test_space_vs_no_space_are_different(self, model):
        """' France' and 'France' must resolve to different token IDs."""
        tid_space = resolve_single_token(model, " France")
        tid_bare = resolve_single_token(model, "France")
        assert tid_space != tid_bare

    def test_invalid_raises(self, model):
        """A multi-token string should raise ValueError with diagnostic info."""
        with pytest.raises(ValueError, match="is not a single token"):
            resolve_single_token(model, "this is definitely multiple tokens")

    def test_error_shows_encoding(self, model):
        """The error message should show what the string actually encodes to."""
        with pytest.raises(ValueError, match="encodes to"):
            resolve_single_token(model, "hello world")
