# pytest cs336-assignment1-basics/tests/test_train_bpe.py
import logging
import os
from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import regex as re
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
import concurrent.futures
from collections import Counter

# from cs336_basics.utils.io import GPT2_PRETOKENIZER_PATTERN


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
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
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


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: bytes(i) for i in range(256)}  # token_id -> token_bytes (str -> encode)
    for i, special_token in enumerate(special_tokens):
        vocab[256 + i] = special_token.encode("utf-8")

    num_processes = 1
    boundaries = _find_chunk_boundaries(input_path, num_processes, b"<|endoftext|>")

    arg_list: list[tuple[int, int, str]] = []
    for i in range(num_processes - 1):
        arg_list.append((boundaries[i], boundaries[i + 1], input_path))

    with Pool(processes=num_processes) as pool:
        results: list[dict[str, int]] = pool.starmap(_pre_tokenize_chunk, arg_list)

    pretoken_cnt: dict[Unknown, Unknown] = {}  #
    for d in results:
        for k, v in d.items():
            t = tuple([bytes(b) for b in k.encode("utf-8")])
            if k in pretoken_cnt:
                pretoken_cnt[t] += v
            else:
                pretoken_cnt[t] = v

    pretoken_indices: dict[Unknown, list[int]] = (
        {}
    )  #  "hello" -> [104, 101, 108, 108, 111]
    merges: dict[tuple[int, int], int] = {}  # (token_id_1, token_id_2) -> new_token_id

    # initialization phase, convert every single byte to a token id
    for pretoken_str, cnt in pretoken_cnt.items():
        token_id_list = list(map(int, pretoken_str.encode("utf-8")))
        pretoken_indices[pretoken_str] = token_id_list
        for token_id in token_id_list:
            vocab[token_id] = token_id.to_bytes(1)

    target_num_merges = vocab_size - len(vocab)

    assert (
        target_num_merges > 0
    ), "Target vocab size must be larger than the number of special tokens"

    while len(vocab) < target_num_merges:
        # (index1, index2) -> count
        # find maximum (index1, index2) pair
        # update the vocab
        # update merges
        counts = defaultdict(int)
        for s, token_ids in pretoken_indices.items():
            for id1, id2 in zip(token_ids[:-1], token_ids[1:]):
                # if is_valid_utf8(bytes([index1, index2])):
                counts[(id1, id2)] += pretoken_cnt[s]

        max_cnt = -1
        max_id12 = None
        for id12, cnt in counts.items():
            if cnt > max_cnt:
                max_id12 = id12
                max_cnt = cnt
            elif cnt == max_cnt:
                if max_id12 is not None and id12 > max_id12:
                    max_id12 = id12

        if max_id12 is not None:
            left_token_id, right_token_id = max_id12
            new_token_id = max(vocab) + 1
            merges[max_id12] = new_token_id
            vocab[new_token_id] = vocab[left_token_id] + vocab[right_token_id]

            for s, token_id_list in pretoken_indices.items():
                pretoken_indices[s] = _replace_sublist(
                    token_id_list,
                    [max_id12[0], max_id12[1]],
                    [new_token_id],
                )

    result_merge = []
    for t in merges:
        result_merge = (t[0].to_bytes(2), t[1].to_bytes(2))
    return (vocab, result_merge)
