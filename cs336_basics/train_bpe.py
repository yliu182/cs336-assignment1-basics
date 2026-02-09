# pytest cs336-assignment1-basics/tests/test_train_bpe.py
import logging
import multiprocessing
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

    with open(input_path, "r") as file:
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
                found_at = mini_chunk.encode("utf-8").find(split_special_token)
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
) -> dict[str, int]:
    """
    Given a file and a list of chunk boundaries, tokenize the file into chunks.
    """
    #  assert isinstance(
    #      special_tokens, list
    #  ), "Must represent special tokens as a list of strings"

    # Read the file in chunks
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk_bytes = file.read(end - start)
        # pretoken_to_cnt = get_pretoken_counts(chunk_bytes)
        input_str = chunk_bytes.decode("utf-8")
        counts = {}
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

    num_processes = 2
    boundaries = _find_chunk_boundaries(input_path, num_processes, b"<|endoftext|>")

    arg_list: list[tuple[int, int, str]] = []
    for i in range(len(boundaries) - 1):
        arg_list.append((boundaries[i], boundaries[i + 1], input_path))

    # Use "fork" start method - this is much more reliable than "spawn" (default on some systems)
    # "fork" copies the parent process memory, avoiding the need to re-import modules
    # which can cause deadlocks especially when running under pytest
    # mp_context = multiprocessing.get_context("fork")
    # with mp_context.Pool(processes=num_processes) as pool:

    with Pool(num_processes) as p:
        results: list[dict[str, int]] = p.starmap(_pre_tokenize_chunk, arg_list)

    pretoken_freq: dict[tuple[bytes, ...], int] = {}
    for d in results:
        for k, v in d.items():
            # Convert string to bytes, then split into individual bytes
            k_bytes = k.encode("utf-8")
            t = tuple(bytes([b]) for b in k_bytes)
            if t in pretoken_freq:
                pretoken_freq[t] += v
            else:
                pretoken_freq[t] = v

    # for key, value in list(pretoken_freq.items())[:3]:
    #     print(f"--- {key}: {value}")

    merges: list[tuple[bytes, bytes]] = []

    # # initialization phase, convert every single byte to a token id
    # for pretoken_str, cnt in pretoken_cnt.items():
    #     token_id_list = list(map(int, pretoken_str.encode("utf-8")))
    #     pretoken_indices[pretoken_str] = token_id_list
    #     for token_id in token_id_list:
    #         vocab[token_id] = token_id.to_bytes(1)

    logging.info("Initializing byte pair frequency table")
    pair_freq = Counter()
    for byte_tuple, cnt in pretoken_freq.items():
        for i in range(len(byte_tuple) - 1):
            pair_freq[(byte_tuple[i], byte_tuple[i + 1])] += cnt

    assert vocab_size > len(
        vocab
    ), "Target vocab size must be larger than the number of special tokens"

    while len(vocab) < vocab_size:
        most_freq_pair = max(pair_freq, key=lambda k: (pair_freq[k], k))
        merges.append(most_freq_pair)

        # Update the vocab
        new_id = max(vocab.keys()) + 1
        vocab[new_id] = b"".join(most_freq_pair)

        # Update the pre-token frequency table and pair frequency table
        new_pretoken_freq = {}

        for pretoken_tuple, cnt in pretoken_freq.items():
            i = 0
            while i < len(pretoken_tuple):
                if pretoken_tuple[i : i + 2] == most_freq_pair:
                    pretoken_tuple, prefix, suffix = _update_byte_tuple(
                        pretoken_tuple, i
                    )

                    pair_freq[most_freq_pair] -= cnt

                    # update pair frequency table
                    if prefix:
                        add_pair = (prefix[-1], vocab[new_id])
                        pair_freq[add_pair] += cnt
                        delete_pair = (prefix[-1], most_freq_pair[0])
                        pair_freq[delete_pair] -= cnt
                    if suffix:
                        add_pair = (vocab[new_id], suffix[0])
                        pair_freq[add_pair] += cnt
                        delete_pair = (most_freq_pair[1], suffix[0])
                        pair_freq[delete_pair] -= cnt

                else:
                    i += 1

            new_pretoken_freq[pretoken_tuple] = cnt

        pretoken_freq = new_pretoken_freq

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
