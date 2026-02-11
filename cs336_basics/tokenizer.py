# uv run pytest cs336-assignment1-basics/tests/test_tokenizer.py
#
# uv run python -m cs336_basics.tokenizer --vocab-path tests/fixtures/gpt2_vocab.json  --merges-path tests/fixtures/gpt2_merges.txt

from argparse import ArgumentParser
from collections.abc import Iterable
from pathlib import Path

import regex as re
from cs336_basics.io import (
    get_tokenizer_from_vocab_merges_path,
    GPT2_PRETOKENIZER_PATTERN,
)
from cs336_basics.train_bpe import train_bpe
from tqdm import tqdm


def _fix_vocab(vocab_i_to_b: dict[int, bytes], vocab_b_to_i: dict[bytes, int]):
    """Make sure all bytes are in the vocab"""
    for i in range(256):
        byte = bytes([i])
        if byte not in vocab_b_to_i:
            vocab_b_to_i[byte] = len(vocab_b_to_i)
            vocab_i_to_b[len(vocab_i_to_b)] = byte
    return dict(int_to_byte=vocab_i_to_b, byte_to_int=vocab_b_to_i)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None,
    ) -> None:
        self.vocab = {}
        self.vocab["int_to_byte"] = vocab
        self.vocab["byte_to_int"] = {v: k for k, v in vocab.items()}
        self.vocab = _fix_vocab(self.vocab["int_to_byte"], self.vocab["byte_to_int"])

        # reorganzie merges into pair -> new token id dict
        self.merges = {}
        for a, b in merges:
            id_pair = (self.vocab["byte_to_int"][a], self.vocab["byte_to_int"][b])
            self.merges[id_pair] = self.vocab["byte_to_int"][a + b]

        # add special tokens as string to id mapping
        self.special_tokens = {}
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in special_tokens:
                token_byte = token.encode("utf-8")
                if token_byte not in self.vocab["byte_to_int"]:
                    self.vocab["byte_to_int"][token_byte] = len(
                        self.vocab["byte_to_int"]
                    )
                    self.vocab["int_to_byte"][
                        len(self.vocab["int_to_byte"])
                    ] = token_byte
                    self.special_tokens[token] = len(self.vocab["int_to_byte"])
                else:
                    self.special_tokens[token] = self.vocab["byte_to_int"][token_byte]

    def merge_token_pair(
        self,
        input_ids: list[int],
        pair: tuple[int, int],
    ) -> list[int]:
        """
        Merges a pair of tokens into a single token.
        """
        if pair not in self.merges:
            return input_ids

        i = 0
        new_input_ids = []
        while i < len(input_ids):
            if i < len(input_ids) - 1 and (input_ids[i], input_ids[i + 1]) == pair:
                merged_id = self.vocab["byte_to_int"][
                    self.vocab["int_to_byte"][pair[0]]
                    + self.vocab["int_to_byte"][pair[1]]
                ]
                new_input_ids.append(merged_id)
                i += 2
            else:
                new_input_ids.append(input_ids[i])
                i += 1
        return new_input_ids

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] = None,
    ):
        vocab, merges = get_tokenizer_from_vocab_merges_path(
            vocab_filepath,
            merges_filepath,
        )

        return cls(vocab, merges, special_tokens)

    def _pretokenization(
        self,
        text: str,
    ) -> list[str]:
        """
        Performs a pre-tokenization step on the input text.

        The purpose of this pretokenization is to split text into meaningful chunks before applying BPE, ensuring that merges don't cross word boundaries inappropriately.
        """
        # text = "She's running 100m! That's fast..."
        # matches = re.findall(GPT2_PRETOKENIZER_PATTERN, text)
        # # Returns: ['She', "'s", ' running', ' 100', 'm', '!', ' That', "'s", ' fast', '...']

        # text = "   spaces   here"
        # matches = re.findall(GPT2_PRETOKENIZER_PATTERN, text)
        # # Returns: ['   ', 'spaces', '   ', 'here']
        return re.findall(GPT2_PRETOKENIZER_PATTERN, text)

    def _encode_chunk(self, text: str) -> list[int]:
        """
        Encodes a string into a list of token ids.
        """
        if text in self.special_tokens:
            return [self.special_tokens[text]]

        text_chunks = self._pretokenization(text)
        output_ids = []
        for chunk in text_chunks:
            input_ids = [
                self.vocab["byte_to_int"][bytes([b])] for b in chunk.encode("utf-8")
            ]
            while len(input_ids) > 1:
                # find earliest pair
                all_pairs = set(zip(input_ids[:-1], input_ids[1:]))
                bigram = min(
                    all_pairs,
                    key=lambda pair: self.merges.get(pair, float("inf")),
                )

                if bigram not in self.merges:
                    break

                input_ids = self.merge_token_pair(input_ids, bigram)
            output_ids.extend(input_ids)
        return output_ids

    def encode(self, text: str, progress_bar: bool = False) -> list[int]:
        """
        Encodes a string into a list of token ids.
        """
        if self.special_tokens:
            special_pattern = (
                "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            )
            special_split_chunk = re.split(special_pattern, text)
        else:
            special_split_chunk = [text]

        ids = []
        for chunk in tqdm(
            special_split_chunk,
            disable=not progress_bar,
            desc=f"Encoding {len(special_split_chunk)} documents",
        ):
            ids += self._encode_chunk(chunk)
        return ids

    def encode_iterable(self, texts: Iterable[str]) -> Iterable[int]:
        """
        Encode the texts into a list of token ids.
        """
        for text in texts:
            ids = self.encode(text)
            for id in ids:
                yield id

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of token ids into a string.
        """
        text_bytes = b"".join([self.vocab["int_to_byte"][i] for i in ids])
        return text_bytes.decode("utf-8", errors="replace")

    def save_vocab_and_merges_to_files(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        vocab_path: str,
        merges_path: str,
    ):
        """
        Save vocab and merges to files in GPT-2 format.

        Args:
            vocab: dict mapping token_id (int) -> token_bytes (bytes)
            merges: list of (bytes, bytes) tuples representing merge operations
            vocab_path: path to save vocab JSON file
            merges_path: path to save merges text file
        """
        # Get the byte-to-unicode mapping (for human-readable output)
        byte_to_unicode = gpt2_bytes_to_unicode()

        # Convert vocab: bytes -> unicode string representation
        # Format: {"token_string": token_id, ...}
        vocab_json = {}
        for token_id, token_bytes in vocab.items():
            # Convert each byte to its unicode representation
            token_str = "".join(byte_to_unicode[b] for b in token_bytes)
            vocab_json[token_str] = token_id

        # Save vocab as JSON
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_json, f, ensure_ascii=False)

        # Save merges as text file (one merge per line, space-separated)
        with open(merges_path, "w", encoding="utf-8") as f:
            for token1, token2 in merges:
                # Convert bytes to unicode string representation
                str1 = "".join(byte_to_unicode[b] for b in token1)
                str2 = "".join(byte_to_unicode[b] for b in token2)
                f.write(f"{str1} {str2}\n")

    @staticmethod
    def load_vocab_and_merges(
        vocab_path: str,
        merges_path: str,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab, merges = get_tokenizer_from_vocab_merges_path(
            vocab_path,
            merges_path,
        )
        return vocab, merges


def parse_args():
    p = ArgumentParser()
    p.add_argument("--vocab-path", required=True)
    p.add_argument("--merges-path", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Theme 1: read vocab and merges from files
    tok = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=["<|endoftext|>"],
    )

    # print(tok.encode("hello world"))
    input_dir = Path(args.data_path)

    # theme 2: just read the raw text, and train the tokenizer from scratch
    # my_vocab, my_merges = train_bpe(
    #     "tests/fixtures/tinystories_sample_5M.txt", 500, ["<|endoftext|>"]
    # )
    # tok = Tokenizer(my_vocab, my_merges, special_tokens=["<|endoftext|>"])
    # print(tok.encode("hello world"))

    # theme 3: calculate compression ratio
    compression_ratios = {}

    #  iterates through all files in a directory that end with "_val.txt" using the glob method with a wildcard pattern.
    for f in input_dir.glob("*_val.txt"):
        # sample 100 docs from each dataset
        ratios = []
        for doc in re.split(tok.split_re, f.read_text())[:200]:
            if doc != "<|endoftext|>" and doc:
                n_bytes = len(doc.encode())
                # print(doc)
                toks = tok.encode(doc)
                # print(toks)
                n_tokens = len(toks)
                ratio = n_bytes / n_tokens
                ratios.append(ratio)
        avg_ratio = np.mean(ratios)
        print(f"{f.name} compression ratio: {avg_ratio:.2f}")
        compression_ratios[f.name] = avg_ratio
    # (input_dir / args.stats_name).write_text(json.dumps(compression_ratios))
    print(compression_ratios)
    # tokenized_path = Path(args.tokenized_data_path)
    # tokenized_path.mkdir(exist_ok=True, parents=True)

    # fpaths = list(input_dir.glob("*train.txt"))
    # if args.include_val_data == 1:
    #     fpaths.extend(list(input_dir.glob("*val.txt")))
    # for fpath in fpaths:
    #     t0 = time.monotonic()
    #     tokens = tok.encode_file(fpath, chunk_size=8 * 1024 * 1024)
    #     taken = time.monotonic() - t0
    #     logger.info(f"Processed {str(fpath)}")
    #     logger.info(f"Took {taken:.1f} s.")
    #     logger.info(
    #         f"Throughput: {fpath.stat().st_size / (1024 * 1024) / taken:.2f} MB/s"
    #     )
    #     fname = fpath.name
    #     np.save(
    #         str((tokenized_path / fname).with_suffix(".npy")),
    #         np.array(tokens, dtype="uint16"),
    #     )
