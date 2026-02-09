# pytest cs336-assignment1-basics/tests/test_train_bpe.py
import logging
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import regex as re
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# from cs336_basics.utils.io import GPT2_PRETOKENIZER_PATTERN
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _find_chunk_boundaries(
    input_path: str,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Open file in BINARY mode since we're working with bytes
    with open(input_path, "rb") as file:
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _get_pretoken_counts(input_bytes: bytes) -> dict[str, int]:
    input_str = bytes.decode("utf-8")
    counts = {}

    with re.finditer(PAT, input_str) as matches:
        for match in matches:
            token = match.group()
            if token in counts:
                counts[token] += 1
            else:
                counts[token] = 1
    return counts


def _pre_tokenize_chunk(
    start: int,
    end: int,
    input_path: str,
    special_tokens: list[str] | None = None,
) -> dict[str, int]:
    """
    Given a file and a list of chunk boundaries, tokenize the file into chunks.
    Special tokens are excluded from tokenization to prevent them from being merged.
    """
    # Read the file in chunks
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk_bytes = file.read(end - start)
        input_str = chunk_bytes.decode("utf-8")

        counts = {}

        # If we have special tokens, we need to split the input on them first
        # to avoid tokenizing parts of special tokens
        if special_tokens:
            # Build a regex pattern that matches any special token
            # Escape special regex characters in the tokens
            # # Let's say these are your special tokens
            # special_tokens = ["<|endoftext|>", "[MASK]", "user.name"]

            # # WITHOUT escaping - these tokens contain special regex characters:
            # # "|" means OR in regex
            # # "[" and "]" define a character class
            # # "." matches any single character

            # # WITH escaping:
            # escaped_tokens = [re.escape(tok) for tok in special_tokens]

            # print("Original tokens:", special_tokens)
            # print("Escaped tokens:", escaped_tokens)
            # Original tokens: ['<|endoftext|>', '[MASK]', 'user.name']
            # Escaped tokens:  ['<\\|endoftext\\|>', '\\[MASK\\]', 'user\\.name']
            escaped_tokens = [re.escape(tok) for tok in special_tokens]
            special_pattern = "|".join(escaped_tokens)

            # Split on special tokens, keeping the delimiters
            # We'll only apply pretokenization to non-special parts
            parts = re.split(f"({special_pattern})", input_str)

            for part in parts:
                if part in special_tokens:
                    # Skip special tokens - don't include them in BPE training
                    continue
                # Apply normal pretokenization to non-special parts
                for match in re.finditer(PAT, part):
                    token = match.group()
                    if token in counts:
                        counts[token] += 1
                    else:
                        counts[token] = 1
        else:
            # No special tokens, just apply normal pretokenization
            for match in re.finditer(PAT, input_str):
                token = match.group()
                if token in counts:
                    counts[token] += 1
                else:
                    counts[token] = 1

    return counts


def _replace_sublist(lst, pattern, replacement):
    # lst = [1, 2, 3, 2, 3, 4]
    # pattern = [2, 3]
    # replace = [7]
    # expected_output = [1, 7, 7, 4]
    result = []
    i = 0
    while i <= len(lst) - len(pattern) + 1:
        if lst[i : i + len(pattern)] == pattern:
            result.extend(replacement)
            i += len(pattern)
        else:
            result.append(lst[i])
            i += 1
    return result


def _update_byte_tuple(byte_tuple: Iterable[bytes], merge_loc: int):
    """
    Merge the byte tuple at the merge location.
    """
    assert len(byte_tuple) > 1, "Cannot merge a byte tuple with length less than 2."
    prefix = byte_tuple[:merge_loc]
    tomerge = byte_tuple[merge_loc : merge_loc + 2]
    suffix = byte_tuple[merge_loc + 2 :]
    new_byte_tuple = prefix + (b"".join(tomerge),) + suffix
    return new_byte_tuple, prefix, suffix


def _pre_tokenize_chunk_wrapper(args: tuple[int, int, str]) -> dict[str, int]:
    """Wrapper to unpack arguments for ProcessPoolExecutor.map()"""
    return _pre_tokenize_chunk(*args)


def _get_first_256_tokens() -> dict[int, bytes]:
    """Return the base vocabulary mapping each byte value (0-255) to its single-byte representation."""
    return {i: bytes([i]) for i in range(256)}


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # vocab = {i: chr(i).encode("utf-8") for i in range(256)}  # token_id -> token_bytes
    vocab = _get_first_256_tokens()
    for i, special_token in enumerate(special_tokens):
        vocab[256 + i] = special_token.encode("utf-8")

    num_processes = 4
    boundaries = _find_chunk_boundaries(input_path, num_processes, b"<|endoftext|>")

    arg_list: list[tuple[int, int, str, list[str] | None]] = []
    for i in range(len(boundaries) - 1):
        arg_list.append((boundaries[i], boundaries[i + 1], input_path, special_tokens))

    with Pool(num_processes) as p:
        results: list[dict[str, int]] = p.starmap(_pre_tokenize_chunk, arg_list)

    # Build pretoken_freq: maps tuple of bytes -> count
    # Also build an index from each pair -> set of pretoken tuples that contain it
    pretoken_freq: dict[tuple[bytes, ...], int] = {}
    for d in results:
        for k, v in d.items():
            k_bytes = k.encode("utf-8")
            t = tuple(bytes([b]) for b in k_bytes)
            if t in pretoken_freq:
                pretoken_freq[t] += v
            else:
                pretoken_freq[t] = v

    merges: list[tuple[bytes, bytes]] = []

    # Initialize pair frequency using a dict for faster operations
    pair_freq: dict[tuple[bytes, bytes], int] = {}
    # Index: pair -> set of pretoken_tuples containing this pair
    pair_to_pretokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    for byte_tuple, cnt in pretoken_freq.items():
        for i in range(len(byte_tuple) - 1):
            pair = (byte_tuple[i], byte_tuple[i + 1])
            if pair in pair_freq:
                pair_freq[pair] += cnt
            else:
                pair_freq[pair] = cnt
            # Build index
            if pair not in pair_to_pretokens:
                pair_to_pretokens[pair] = set()
            pair_to_pretokens[pair].add(byte_tuple)

    assert vocab_size > len(
        vocab
    ), "Target vocab size must be larger than the number of special tokens"

    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        if not pair_freq:
            break

        # Find the most frequent pair (with lexicographic tie-breaking)
        most_freq_pair = max(pair_freq, key=lambda k: (pair_freq[k], k))

        # Skip if count is 0 or negative
        if pair_freq[most_freq_pair] <= 0:
            # Clean up zero/negative entries and try again
            pair_freq = {k: v for k, v in pair_freq.items() if v > 0}
            if not pair_freq:
                break
            most_freq_pair = max(pair_freq, key=lambda k: (pair_freq[k], k))

        merges.append(most_freq_pair)
        new_token = b"".join(most_freq_pair)
        new_id = len(vocab)
        vocab[new_id] = new_token

        # Only update pretokens that contain this pair
        affected_pretokens = pair_to_pretokens.get(most_freq_pair, set()).copy()

        for old_tuple in affected_pretokens:
            if old_tuple not in pretoken_freq:
                continue

            cnt = pretoken_freq[old_tuple]

            # Build new tuple by merging all occurrences of the pair
            new_list = []
            i = 0
            while i < len(old_tuple):
                if (
                    i < len(old_tuple) - 1
                    and (old_tuple[i], old_tuple[i + 1]) == most_freq_pair
                ):
                    # Before merge: update frequencies for adjacent pairs
                    # Left neighbor loses old pair, gains new pair
                    if new_list:
                        old_left_pair = (new_list[-1], most_freq_pair[0])
                        pair_freq[old_left_pair] = pair_freq.get(old_left_pair, 0) - cnt
                        new_left_pair = (new_list[-1], new_token)
                        pair_freq[new_left_pair] = pair_freq.get(new_left_pair, 0) + cnt

                    # Right neighbor: look ahead
                    if i + 2 < len(old_tuple):
                        old_right_pair = (most_freq_pair[1], old_tuple[i + 2])
                        pair_freq[old_right_pair] = (
                            pair_freq.get(old_right_pair, 0) - cnt
                        )
                        new_right_pair = (new_token, old_tuple[i + 2])
                        pair_freq[new_right_pair] = (
                            pair_freq.get(new_right_pair, 0) + cnt
                        )

                    # Decrease count for the merged pair
                    pair_freq[most_freq_pair] = pair_freq.get(most_freq_pair, 0) - cnt

                    new_list.append(new_token)
                    i += 2
                else:
                    new_list.append(old_tuple[i])
                    i += 1

            new_tuple = tuple(new_list)

            # Update pretoken_freq
            del pretoken_freq[old_tuple]
            if new_tuple in pretoken_freq:
                pretoken_freq[new_tuple] += cnt
            else:
                pretoken_freq[new_tuple] = cnt

            # Update pair_to_pretokens index
            # Remove old_tuple from all its pairs
            for j in range(len(old_tuple) - 1):
                old_pair = (old_tuple[j], old_tuple[j + 1])
                if (
                    old_pair in pair_to_pretokens
                    and old_tuple in pair_to_pretokens[old_pair]
                ):
                    pair_to_pretokens[old_pair].discard(old_tuple)

            # Add new_tuple to all its pairs
            for j in range(len(new_tuple) - 1):
                new_pair = (new_tuple[j], new_tuple[j + 1])
                if new_pair not in pair_to_pretokens:
                    pair_to_pretokens[new_pair] = set()
                pair_to_pretokens[new_pair].add(new_tuple)

        # Clean up the merged pair from index
        if most_freq_pair in pair_to_pretokens:
            del pair_to_pretokens[most_freq_pair]
        if most_freq_pair in pair_freq:
            del pair_freq[most_freq_pair]

    return (vocab, merges)
    # for i in range(len(pretoken_tuple)):
    #     if t[i : i + 2] == most_freq_pair:
    #         prefix, suffix, new_bytes = merge_byte_string()

    # (index1, index2) -> count
    # find maximum (index1, index2) pair
    # update the vocab
    # update merges
    # counts = defaultdict(int)
    # for s, token_ids in pretoken_indices.items():
    #     for id1, id2 in zip(token_ids[:-1], token_ids[1:]):
    #         # if is_valid_utf8(bytes([index1, index2])):
    #         counts

    # while len(vocab) < target_num_merges:
    # (index1, index2) -> count
    # find maximum (index1, index2) pair
    # update the vocab
    # update merges
    # counts = defaultdict(int)
    # for s, token_ids in pretoken_indices.items():
    #     for id1, id2 in zip(token_ids[:-1], token_ids[1:]):
    #         # if is_valid_utf8(bytes([index1, index2])):
    #         counts[(id1, id2)] += pretoken_cnt[s]

    # max_cnt = -1
    # max_id12 = None
    # for id12, cnt in counts.items():
    #     if cnt > max_cnt:
    #         max_id12 = id12
    #         max_cnt = cnt
    #     elif cnt == max_cnt:
    #         if max_id12 is not None and id12 > max_id12:
    #             max_id12 = id12

    # if max_id12 is not None:
    #     left_token_id, right_token_id = max_id12
    #     new_token_id = max(vocab) + 1
    #     merges[max_id12] = new_token_id
    #     vocab[new_token_id] = vocab[left_token_id] + vocab[right_token_id]

    #     for s, token_id_list in pretoken_indices.items():
    #         pretoken_indices[s] = _replace_sublist(
    #             token_id_list,
    #             [max_id12[0], max_id12[1]],
    #             [new_token_id],
    #         )

    # for t in merges:
    #     result_merge.append((t[0].to_bytes(2), t[1].to_bytes(2)))
