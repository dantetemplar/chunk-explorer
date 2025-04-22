import inspect
import itertools
import json
import typing
from typing import Any, TypedDict

import streamlit as st
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter,
    TokenTextSplitter,
)
from transformers import AutoTokenizer

from texts import TEXT_HTML, TEXT_MARKDOWN, TEXT_PLAIN
from wordllama_adapter import WordLlamaTextSplitter


class Splitter(TypedDict):
    class_: type
    kwargs: dict[str, Any]
    name: str
    showcase_text: str


SPLITTERS: list[Splitter] = [
    {
        "class_": RecursiveCharacterTextSplitter,
        "kwargs": {"chunk_size": 100, "chunk_overlap": 0, "strip_whitespace": False, "add_start_index": True},
        "name": "🦜🔗 RecursiveCharacterTextSplitter",
        "showcase_text": TEXT_PLAIN,
    },
    {
        "class_": CharacterTextSplitter,
        "kwargs": {
            "chunk_size": 100,
            "chunk_overlap": 0,
            "strip_whitespace": False,
            "keep_separator": True,
            "add_start_index": True,
        },
        "name": "🦜🔗 CharacterTextSplitter",
        "showcase_text": TEXT_PLAIN,
    },
    {
        "class_": TokenTextSplitter,
        "kwargs": {"chunk_size": 50, "chunk_overlap": 0, "encoding_name": "cl100k_base", "add_start_index": True},
        "name": "🦜🔗 TokenTextSplitter",
        "showcase_text": TEXT_PLAIN,
    },
    {
        "class_": MarkdownHeaderTextSplitter,
        "kwargs": {
            "headers_to_split_on": [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
            "strip_headers": False,
        },
        "name": "🦜🔗 MarkdownHeaderTextSplitter",
        "showcase_text": TEXT_MARKDOWN,
    },
    {
        "class_": HTMLHeaderTextSplitter,
        "kwargs": {"headers_to_split_on": [("h1", "Main Topic"), ("h2", "Sub Topic")]},
        "name": "🦜🔗 HTMLHeaderTextSplitter",
        "showcase_text": TEXT_HTML,
    },
    {
        "class_": NLTKTextSplitter,
        "kwargs": {"chunk_size": 100, "chunk_overlap": 0, "strip_whitespace": False},
        "name": "🦜🔗 NLTKTextSplitter",
        "showcase_text": TEXT_PLAIN,
    },
    {
        "class_": SpacyTextSplitter,
        "kwargs": {"chunk_size": 100, "chunk_overlap": 0, "strip_whitespace": False},
        "name": "🦜🔗 SpacyTextSplitter",
        "showcase_text": TEXT_PLAIN,
    },
    {
        "class_": WordLlamaTextSplitter,
        "kwargs": {
            "target_size": 200,
            "intermediate_size": 20,
            "cleanup_size": 5,
        },
        "name": "🦙 WordLlamaTextSplitter",
        "showcase_text": TEXT_PLAIN,
    },
]
"""
Splitters from https://python.langchain.com/api_reference/text_splitters/index.html
"""


@st.cache_resource
def tokenizers():
    return {
        "cl100k_base": tiktoken.get_encoding("cl100k_base"),
        "yandex": AutoTokenizer.from_pretrained("yandex/YandexGPT-5-Lite-8B-instruct"),
    }


TOKENIZERS = tokenizers()


def type_equals(a: Any, b: Any) -> bool:
    if a == b:
        return True

    origin_a = typing.get_origin(a) or a
    origin_b = typing.get_origin(b) or b
    if origin_a == origin_b:
        args_a = typing.get_args(a)
        args_b = typing.get_args(b)
        # Treat generic with and without args as equal
        if (not args_a and args_b) or (args_a and not args_b):
            return True
        if len(args_a) == len(args_b):
            return all(type_equals(x, y) for x, y in zip(args_a, args_b))
        return False
    return False


def create_widget_for_param(
    name: str, param_type: Any, default: Any, length_functions: dict[str, Any], disabled: bool = False, original_type: Any = None
) -> Any:
    """Create a Streamlit widget for a parameter based on its type."""

    help_type = original_type or param_type
    if name == "length_function":
        # Special handling for length_function
        func_value = st.selectbox(
            name,
            options=list(length_functions.keys()),
            index=list(length_functions.keys()).index("len"),
            help=f"Function to measure the length of text chunks (type: {help_type})",
            disabled=disabled,
        )
        if func_value:
            return length_functions[func_value]
        return None
    elif name == "add_start_index":  # always add start index
        return st.checkbox(
            name, value=True, help=f"Always add start index to the chunk metadata (type: {help_type})", disabled=True
        )
    elif param_type is bool:
        return st.checkbox(
            name, value=default if default is not None else False, help=str(help_type), disabled=disabled
        )
    elif param_type is str:
        value = st.text_input(
            name,
            value=json.dumps(default) if default is not None else None,
            help=str(help_type),
            disabled=disabled,
            placeholder='"Hello world" or "\\n"',
        )
        if value:
            return json.loads(value)
        return None
    elif param_type is int:
        int_value = st.number_input(name, value=default, step=1, help=str(help_type), disabled=disabled, min_value=0)
        if int_value is not None:
            return int(int_value)
        return None
    elif param_type is float:
        float_value = st.number_input(name, value=default, help=str(help_type), disabled=disabled, min_value=0.0)
        if float_value is not None:
            return float(float_value)
    elif type_equals(param_type, list[str]):
        if name == "separators":
            separator_input = st.text_input(
                name,
                value=json.dumps(default) if default is not None else None,
                help=f"Enter separators as json array ({help_type})",
                disabled=disabled,
                placeholder='[" ", "\\n\\n", "\\n"]',
            )
            if separator_input:
                return json.loads(separator_input)
            return None
        else:
            text_input = st.text_input(
                name,
                value=json.dumps(default) if default is not None else None,
                help=f"Enter values as json array ({help_type})",
                disabled=disabled,
                placeholder='["Hello", "World"]',
            )
            if text_input:
                return json.loads(text_input)
            return None

    elif typing.get_origin(param_type) is typing.Union and type(None) in typing.get_args(param_type):
        # make two columns, one with toggle for None, one with the type
        toggle_col, type_col = st.columns([1, 2])
        with toggle_col:
            toggle_value = st.checkbox(name, value=default is not None, help=str(help_type))

        is_none = not toggle_value

        with type_col:
            # exclude None from the type
            new_args = [t for t in typing.get_args(param_type) if t is not type(None)]
            new_type = typing.get_origin(param_type)[*new_args]
            value = create_widget_for_param(name, new_type, default, length_functions, disabled=is_none, original_type=param_type)
        if is_none:
            return None
        if value:
            return value
        return None
    elif typing.get_origin(param_type) is typing.Union:
        args = typing.get_args(param_type)

        # if all args are Literal, use a selectbox
        all_literals = True
        options: list[Any] = []
        for arg in args:
            if typing.get_origin(arg) is typing.Literal:
                options.extend(typing.get_args(arg))
            elif arg is bool:
                options.append(True)
                options.append(False)
            else:
                all_literals = False
                break

        if all_literals:
            return st.selectbox(
                name,
                options=options,
                index=options.index(default) if default is not None else 0,
                format_func=json.dumps,
                help=str(help_type),
                disabled=disabled,
            )

        # typing.Union[typing.Literal['all'], typing.AbstractSet[str]]
        # typing.Union[typing.Literal['all'], typing.Collection[str]]])
        if type_equals(param_type, typing.Union[typing.Literal["all"], typing.AbstractSet[str]]) or type_equals(
            param_type, typing.Union[typing.Literal["all"], typing.Collection[str]]
        ):
            # text input for the set
            set_input = st.text_input(
                name,
                value=json.dumps(list(default) if isinstance(default, set) else default)
                if default is not None
                else None,
                help=str(help_type),
                disabled=disabled,
            )
            if set_input:
                r = json.loads(set_input)
                if isinstance(r, list):
                    return set(r)
                return r
            return None

    # For unknown types, just display the type
    print(f"Unknown type: {param_type}, default: {default}")
    st.text_input(name, value=str(default), help=str(param_type), disabled=True)
    return default

def adapt_langchain_splitter_to_streamlit(splitter: Splitter) -> dict:
    """Get user inputs for splitter parameters."""
    splitter_class = splitter["class_"]
    chunkervis_defaults = splitter["kwargs"]

    # Define available length functions
    length_functions = {
        "len": len,
        "count_words": lambda x: len(x.split()),
        "count_chars": lambda x: len(x.replace(" ", "")),
        "count_lines": lambda x: len(x.splitlines()),
    }

    params: dict[str, Any] = {}
    defaults: dict[str, Any] = {**chunkervis_defaults}
    call_defaults: dict[str, Any] = {}

    # Get type hints and default values for the class
    type_hints = typing.get_type_hints(splitter_class.__init__)
    for name, param in inspect.signature(splitter_class.__init__).parameters.items():
        if name not in type_hints:
            type_hints[name] = param.annotation
        if param.default is not inspect.Parameter.empty:
            if name not in defaults:
                defaults[name] = param.default
            if name not in call_defaults:
                call_defaults[name] = param.default

    # and its base classes if **kwargs
    if "kwargs" in type_hints:
        for base in splitter_class.__bases__:
            for k, v in typing.get_type_hints(base.__init__).items():
                if k not in type_hints:
                    type_hints[k] = v
            for name, param in inspect.signature(base.__init__).parameters.items():
                if name not in type_hints:
                    type_hints[name] = param.annotation
                if param.default is not inspect.Parameter.empty:
                    if name not in defaults:
                        defaults[name] = param.default
                    if name not in call_defaults:
                        call_defaults[name] = param.default
    type_hints.pop("self", None)
    type_hints.pop("kwargs", None)
    type_hints.pop("return", None)

    # Create columns for compact layout
    cols = st.columns(2)
    col_idx = 0

    # Sort type_hits such way:
    # - chunk_*** parameters should be first
    # - booleans should be in the end
    def key(x):
        if x[0].startswith("chunk_"):
            return 0, x[0]
        else:
            return 1, x[0]

    sorted_type_hints = list(sorted(type_hints.items(), key=key))
    booleans = [x for x in sorted_type_hints if x[1] is bool]
    sorted_type_hints = [x for x in sorted_type_hints if x[1] is not bool]
    # First display all non-boolean parameters
    for name, param_type in sorted_type_hints:
        # Get default value: first try SPLITTERS, then signature default
        default = defaults.get(name)
        # Use alternating columns
        with cols[col_idx]:
            value = create_widget_for_param(name, param_type, default, length_functions)
            if value is not None:
                params[name] = value
        col_idx = (col_idx + 1) % 2

    booleans_cols = st.columns(5)
    col_idx = 0
    # Then display all boolean in 5 columns
    for name, param_type in booleans:
        with booleans_cols[col_idx]:
            value = create_widget_for_param(name, param_type, defaults.get(name), length_functions)
            if value is not None:
                params[name] = value
        col_idx = (col_idx + 1) % 5

    return params


def find_longest_suffix_prefix_match(text1: str, text2: str) -> tuple[int, int]:
    """Find the longest suffix of text1 that matches a prefix of text2.

    Returns:
        tuple[int, int]: (left_overlap, right_overlap) where:
            - left_overlap: negative number indicating how much text1 overlaps with text2
            - right_overlap: positive number indicating how much text2 overlaps with text1
    """
    max_match = 0
    for i in range(1, min(len(text1), len(text2)) + 1):
        if text1[-i:] == text2[:i]:
            max_match = i

    if max_match == 0:
        return (0, 0)

    return (-max_match, max_match)


def identify_overlaps(
    splitted_documents: list[Document], chunk_size: int | None = None, chunk_overlap: int | None = None
):
    """Identify overlaps in the splitted documents (when next chunk started with the end of the previous chunk)."""

    overlaps: dict[int, tuple[int, int]] = {}

    for i, (doc1, doc2) in enumerate(itertools.pairwise(splitted_documents)):
        s1 = doc1.metadata.get("start_index")
        s2 = doc2.metadata.get("start_index")
        len_1 = len(doc1.page_content)

        if s1 is None or s2 is None:
            continue
        if s1 == -1 or s2 == -1:
            # Fallback on finding biggest suffix of doc1 that is prefix of doc2
            left, right = find_longest_suffix_prefix_match(doc1.page_content, doc2.page_content)
            if left < 0:  # Only record if there is an actual overlap
                overlaps[i] = (left, right)
            continue

        assert s2 >= s1, f"s2 must be greater than s1, but s1={s1}, s2={s2}"
        left = s2 - s1 - len_1
        right = s1 - s2 + len_1

        if left >= 0:  # should be negative to be an overlap
            continue

        overlaps[i] = (left, right)

    return overlaps


if __name__ == "__main__":
    splitter = TokenTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        encoding_name="cl100k_base",
    )
    chunks = splitter.create_documents([TEXT_PLAIN])

    print(chunks)
