import unittest

from mlx_audio.tts.models.qwen3_tts.text_commit import (
    CommitPolicy,
    TextCommitError,
    TextCommitter,
    TextInputLimitError,
    TokenEncoding,
    TokenizationMovedCommittedBoundaryError,
)


def tokenize_words(text: str) -> TokenEncoding:
    """Tiny deterministic tokenizer with text-relative offsets."""
    ids, offsets = [], []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        start = index
        while index < len(text) and not text[index].isspace():
            index += 1
        if start != index:
            ids.append(sum(map(ord, text[start:index])))
            offsets.append((start, index))
    return TokenEncoding(tuple(ids), tuple(offsets))


class TestTextCommitter(unittest.TestCase):
    def test_safe_word_keeps_trailing_word_and_its_prefix_space(self):
        committer = TextCommitter(tokenize_words)
        committer.append("Hello ")
        self.assertEqual(committer.snapshot.sealed_token_ids, ())
        self.assertEqual(committer.snapshot.unstable_suffix, "Hello ")

        committer.append("world ")
        self.assertEqual(committer.snapshot.sealed_token_ids, (500,))
        self.assertEqual(committer.snapshot.unstable_suffix, " world ")

    def test_finalize_flushes_all_tokens(self):
        committer = TextCommitter(tokenize_words)
        committer.append("Hello world")
        self.assertEqual(committer.snapshot.sealed_token_ids, (500,))
        result = committer.finalize()
        self.assertEqual(result.newly_sealed_tokens, 1)
        self.assertEqual(committer.consume_ready(10), (500, 552))

    def test_append_is_atomic_when_tokenizer_fails(self):
        def tokenizer(text):
            if text.endswith("!"):
                raise ValueError("synthetic tokenizer failure")
            return tokenize_words(text)

        committer = TextCommitter(tokenizer)
        committer.append("Hello world ")
        before = committer.snapshot
        with self.assertRaises(ValueError):
            committer.append("!")
        self.assertEqual(committer.snapshot, before)

    def test_moved_sealed_prefix_is_rejected_atomically(self):
        def tokenizer(text):
            if text == "hello world ":
                return TokenEncoding((1,), ((0, 5),))
            return tokenize_words(text)

        committer = TextCommitter(tokenizer)
        committer.append("hello world ")
        before = committer.snapshot
        with self.assertRaises(TokenizationMovedCommittedBoundaryError):
            committer.append("again ")
        self.assertEqual(committer.snapshot, before)

    def test_safe_sentence_waits_for_strong_boundary(self):
        committer = TextCommitter(tokenize_words, policy=CommitPolicy.SAFE_SENTENCE)
        committer.append("One sentence. Next")
        self.assertEqual(committer.snapshot.unstable_suffix, " Next")
        self.assertEqual(committer.snapshot.sealed_char_end, len("One sentence."))

    def test_input_limit_does_not_mutate_snapshot(self):
        committer = TextCommitter(tokenize_words, max_input_chars=5)
        before = committer.snapshot
        with self.assertRaises(TextInputLimitError):
            committer.append("abcdef")
        self.assertEqual(committer.snapshot, before)

    def test_empty_append_is_idempotent(self):
        committer = TextCommitter(tokenize_words)
        before = committer.snapshot
        result = committer.append("")
        self.assertEqual(committer.snapshot, before)
        self.assertEqual(result.accepted_chars, 0)

    def test_prepare_does_not_mutate_until_commit(self):
        committer = TextCommitter(tokenize_words)
        proposal = committer.prepare_append("Hello world ")
        self.assertEqual(committer.snapshot.raw_text, "")
        committer.commit(proposal)
        self.assertEqual(committer.snapshot.raw_text, "Hello world ")

    def test_stale_proposal_is_rejected(self):
        committer = TextCommitter(tokenize_words)
        proposal = committer.prepare_append("Hello ")
        committer.append("Other ")
        with self.assertRaises(TextCommitError):
            committer.commit(proposal)


if __name__ == "__main__":
    unittest.main()
