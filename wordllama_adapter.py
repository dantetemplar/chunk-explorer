import streamlit as st
from langchain_core.documents import Document
from wordllama import WordLlama, WordLlamaInference


@st.cache_resource
def load_wordllama() -> WordLlamaInference:
    return WordLlama.load()


class WordLlamaTextSplitter:
    """Split text into semantically coherent chunks.

    Args:
        text (str): The input text to split.
        target_size (int, optional): Desired size of text chunks. Defaults to 1536.
        window_size (int, optional): Window size for similarity matrix averaging. Defaults to 3.
        poly_order (int, optional): Polynomial order for Savitzky-Golay filter. Defaults to 2.
        savgol_window (int, optional): Window size for Savitzky-Golay filter. Defaults to 3.
        cleanup_size (int, optional): Size for cleanup operations. Defaults to 24.
        intermediate_size (int, optional): Intermediate size for initial splitting. Defaults to 96.
        return_minima (bool, optional): If True, return the indices of minima instead of chunks. Defaults to False.

    Returns:
        List[str]: List of semantically split text chunks.
    """

    def __init__(
        self,
        target_size: int = 1536,
        window_size: int = 3,
        poly_order: int = 2,
        savgol_window: int = 3,
        cleanup_size: int = 24,
        intermediate_size: int = 96,
        return_minima: bool = False,
        add_start_index: bool = True,
    ):
        #     target_size: int = 1536,
        # window_size: int = 3,
        # poly_order: int = 2,
        # savgol_window: int = 3,
        # cleanup_size: int = 24,
        # intermediate_size: int = 96,
        # return_minima: bool = False

        self.target_size = target_size
        self.window_size = window_size
        self.poly_order = poly_order
        self.savgol_window = savgol_window
        self.cleanup_size = cleanup_size
        self.intermediate_size = intermediate_size
        self.return_minima = return_minima
        self.add_start_index = add_start_index
        self.wl = load_wordllama()

    def create_documents(self, texts: list[str]) -> list[Document]:
        documents = []
        for text in texts:
            chunks = self.wl.split(
                text,
                target_size=self.target_size,
                window_size=self.window_size,
                poly_order=self.poly_order,
                savgol_window=self.savgol_window,
                cleanup_size=self.cleanup_size,
                intermediate_size=self.intermediate_size,
                return_minima=self.return_minima,
            )
            current_pos = 0
            for chunk in chunks:
                # Find the actual start position of this chunk in the original text
                start_pos = text.find(chunk, current_pos)
                if start_pos == -1:
                    # If chunk not found, try from beginning (might be an overlap)
                    start_pos = text.find(chunk)
                    if start_pos == -1:
                        # If still not found, use current_pos as fallback
                        start_pos = current_pos

                metadata = {"start_index": start_pos} if self.add_start_index else {}
                documents.append(Document(page_content=chunk, metadata=metadata))
                current_pos = start_pos + len(chunk)
        return documents
