import tiktoken
from chonkie import TokenChunker

from texts import TEXT_PLAIN

chunker = TokenChunker(tokenizer=tiktoken.get_encoding("cl100k_base"), chunk_size=100, chunk_overlap=10)

for chunk in chunker.chunk(TEXT_PLAIN):
    print(chunk.start_index)