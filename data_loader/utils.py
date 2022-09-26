from lib2to3.pgen2 import token
import os
import itertools
import numpy as np

from typing import List, Tuple
from tqdm import tqdm
from time import perf_counter as pc
from bashplotlib.histogram import plot_hist


def __batch(iterable, n=1):
    """
    Args:
        iterable : iterable list
        n : batch size as many as you want split
    """
    if isinstance(iterable, list):
        pass
    else:
        raise TypeError("Check if loaded data is list type")

    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def __get_token_length(tokenizer, text_data: List[str], batch_size: int = 256, add_special_tokens: bool = True):
    """
    텍스트 리스트를 입력 받아 토크나이징 후 토큰 index 리스트와 토큰의 길이 쌍으로 반환하는 함수
    Args :
        tokenizer : 토크나이저
        input_data(List[str]) : 토큰화할 텍스트 데이터 리스트
        batch_size(int) : 제너레이터 배치사이즈
        add_special_tokens : 토큰화할 때 스페셜 토큰 포함 여부
    Returns :
        input_data(List[str]) : 입력 text 데이터 list(==text_data)
        token_lens(List[int]) : 입력 데이터의 토큰의 size를 튜플로 갖는 리스트
    """
    result = [(s, [len(tokens) for tokens in tokenizer(s, add_special_tokens=True).input_ids]) for s in __batch(text_data, batch_size)]
    return list(itertools.chain(*[item[0] for item in result])), list(itertools.chain(*[item[1] for item in result]))


def packing_data(tokenizer, src_data, tgt_data, batch_size=256, max_token_length=256, merge_direction="bidirection"):
    """
    A function that sorts the sentences in short order and merges them within the max token length size.
    Args:
        tokenizer : mBart50 Tokenizer
        src_data : Loaded source data from with open function, ex) ['src1\n', 'src2\n',...]
        tgt_data : Loaded target data from with open function, ex) ['tgt1\n', 'tgt2\n',...]
        batch_size : Batch size as many as you want split
        max_token_length : Limited number of tokens when sent is tokenized from the called tokenizer function
    Returns:
        packed_src : List of List[str] packed by batch size, ex) [['src1\n', 'src2\n',...],['src10\n', 'src20\n',...],...]
        packed_tgt : List of List[str] packed by batch size, ex) [['src1\n', 'src2\n',...],['src10\n', 'src20\n',...],...]
        packed_len : Full-length list of packed sentences
    """
    start = pc()
    src_data, src_lens = __get_token_length(tokenizer=tokenizer, text_data=src_data, batch_size=batch_size)
    tgt_data, tgt_lens = __get_token_length(tokenizer=tokenizer, text_data=tgt_data, batch_size=batch_size)
    end = pc()

    print(f"Elapsed time for tokenizing batched src & tgt data : {end-start}")
    print(f"Max length : {max(src_lens)}")

    assert len(src_data) == len(tgt_data) == len(src_lens) == len(tgt_lens)
    print("Distribution of source sentense's len")
    plot_hist(src_lens, bincount=100)
    print("Distribution of target sentense's len")
    plot_hist(tgt_lens, bincount=100)

    parallel_data = sorted(zip(src_data, tgt_data, src_lens, tgt_lens), key=lambda item: item[2], reverse=False)
    print(f"Len of parallel data : {len(parallel_data)}")

    trigger, src_len, tgt_len = 0, 0, 0
    packed_src, packed_tgt, packed_len = [], [], []
    joined_src, joined_tgt = [], []

    if merge_direction in ["unidirection", "bidirection"]:
        if merge_direction == "bidirection":
            # uniform mode
            print(f"Merge datas in {merge_direction}")
            lens = [max(item) for item in zip(src_lens, tgt_lens)]
            merge_list, over_cnt = merge_data_by_limit(lens, limit=max_token_length)
            print(f"over cnt {over_cnt}")
            for item in merge_list:
                merge_src = list()
                merge_tgt = list()
                merge_token_num_src = 0
                merge_token_num_tgt = 0
                for i in item:
                    merge_src.append(src_data[i])
                    merge_tgt.append(tgt_data[i])
                    merge_token_num_src += src_lens[i]
                    merge_token_num_tgt += tgt_lens[i]
                packed_src.append(merge_src)
                packed_tgt.append(merge_tgt)
                packed_len.append([merge_token_num_src, merge_token_num_tgt])
            print("bi",len(packed_src))
            
        elif merge_direction == "unidirection":
            print(f"Merge datas in {merge_direction}")
            for src, tgt, src_token_num, tgt_token_num in tqdm(parallel_data, total=len(parallel_data)):
                sent_len = src_token_num if src_token_num > tgt_token_num else tgt_token_num

                if trigger + sent_len > max_token_length:
                    packed_src.append(joined_src)
                    packed_tgt.append(joined_tgt)
                    packed_len.append([src_len, tgt_len])

                    joined_src, joined_tgt = [], []
                    joined_src.append(src)
                    joined_tgt.append(tgt)
                    src_len, tgt_len = src_token_num, tgt_token_num

                    trigger = sent_len
                else:
                    joined_src.append(src)
                    joined_tgt.append(tgt)
                    src_len += src_token_num
                    tgt_len += tgt_token_num
                    trigger += sent_len
            print("uni",len(packed_src))
    else:
        raise ValueError(f"Check merge direction type : {merge_direction}")

    print(f"Packed efficiency : {np.array(packed_len).mean(axis=0)} / {np.array(packed_len).mean(axis=0)/max_token_length}")
    print(f"overed cnt : {len([c for c in packed_len if c[0] > 256])}")
    example = [c for c in packed_len if c[0] <= 256]
    print(f"Inbound sent's mean length : {np.array(example).mean(axis=0)}")
    print(f"Len of packed data : {len(packed_src)}, {len(packed_tgt)}")
    print(f"*****Quantile of packing sents : {np.quantile([item[0] for item in packed_len], [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])}*****")

    plot_hist([item[0] for item in packed_len], bincount=100)

    return packed_src, packed_tgt, packed_len


def bisect_left(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
    insert just before the leftmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if key(a[mid]) < x:
                lo = mid + 1
            else:
                hi = mid
    return lo


def __get_lengths_by_inputs(input_data: List[str]):
    """
    문장 리스트를 받아서 각 문장의 길이 리스트를 반환하는 함수
    Args:
        input_data(list) : 입력 문장 리스트
    Returns:
        length_list(list) : 입력 문장들의 길이 리스트
    """
    return [len(data) for data in input_data]


def __get_histogram_by_length(input_data: List[str]):
    """
    입력 리스트의 원소의 길이로 히스토그램을 반환하는 함수
    Args:
        input_data(list) : 입력 문장 리스트
    Returns:
        hist(array) : 빈도 리스트
        bin_edges(array) : bin 경계 리스트
    """
    length_list = __get_lengths_by_inputs(input_data)
    return np.histogram(np.array(length_list))


def merge_data_by_limit(input_data: List[int], limit: int):
    """
    int 리스트를 받아 기준 숫자를 넘지 않도록 제한 병합 수 만큼 원소를 병합하는 함수
    Args:
        input_data(list[int]) : 입력 리스트
        limit : 병합된 원소들의 합의 제한
    Returns:
        merge_list(list[list[int]]) : 기준에 따라 병합된 리스트. 원소는 origin 리스트의 인덱스
        over_cnt(int) : limit을 넘는 데이터의 개수
    """

    sorted_list = sorted(__get_index_tuple_by_list(input_data), key=lambda x: x[1])
    over_idx = bisect_left(sorted_list, limit - sorted_list[0][1], key=lambda x: x[1])

    merge_list = [[idx] for idx, _ in sorted_list[over_idx:]]

    rigth_idx = 0
    left_idx = over_idx - 1

    def get_origin_idx(idx):
        return sorted_list[idx][0]

    def get_origin_num(idx):
        return sorted_list[idx][1]

    sum_idxs = [get_origin_idx(left_idx)]
    sum_num = get_origin_num(left_idx)

    while left_idx > rigth_idx:
        if sum_num + get_origin_num(rigth_idx) >= limit:
            merge_list.append(sum_idxs)
            left_idx -= 1
            sum_num = get_origin_num(left_idx)
            sum_idxs = [get_origin_idx(left_idx)]
        else:
            sum_idxs.append(get_origin_idx(rigth_idx))
            sum_num += get_origin_num(rigth_idx)
            rigth_idx += 1
    if sum_idxs:
        merge_list.append(sum_idxs)

    return merge_list, len(sorted_list) - over_idx


def __get_index_tuple_by_list(input_data: List[Tuple]):
    return [(index, item) for index, item in enumerate(input_data)]
