import json
import time
from html import escape
from pprint import pprint

import chonkie
import streamlit as st
import torch
from langchain_core.documents import Document
from streamlit.components.v1 import html

torch.classes.__path__ = []
st.set_page_config(layout="wide", page_title="Chunking Strategies", page_icon=":scissors:")

from strategies import SPLITTERS, TOKENIZERS, adapt_langchain_splitter_to_streamlit, identify_overlaps  # noqa: E402
from texts import TEXT_HTML, TEXT_JS, TEXT_MARKDOWN, TEXT_PLAIN, TEXT_PYTHON  # noqa: E402

# Add a container for metadata display
metadata_container = st.empty()

st.markdown(
    """
<style>
/* Notification styles */
.notification {
  position: fixed;
  bottom: 20px;
  right: 20px;
  background-color: white;
  border: 1px solid #ccc;
  border-radius: 4px;
  padding: 10px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  max-width: 400px;
  max-height: 300px;
  overflow: auto;
  z-index: 1000;
  display: none;
}

.notification pre {
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: monospace;
  font-size: 0.9em;
  color: black;
}

.notification:hover {
    cursor: default;
}

.close-button {
  position: absolute;
  top: 5px;
  right: 5px;
  background: none;
  border: none;
  font-size: 1.2em;
  cursor: pointer;
  color: #666;
  padding: 0 5px;
}

.close-button:hover {
  color: #333;
}

/* Override Streamlit's subheader to be smaller */
h3 {
    font-size: 1.25rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

chunk_container_styles = """
<style>
.chunk-container {
  flex: 1; /* Take all remaining space */
  padding: 1rem;
  margin: 1rem auto;
  border: 1px solid #ccc;
  background-color: #fafafa;
  font-family: monospace;      /* Monospace helps visualize spacing */
  font-size: 1rem;
}

/* Style each chunk inside the container */
.chunk {
  white-space: pre-wrap;       /* Also preserve inside each chunk */
  vertical-align: top;
  transition: all 0.2s ease;   /* Smooth transition for hover effects */
  border-radius: 4px;         /* Rounded corners */
  line-height: 1.8;           /* Increased line spacing */
}

/* Three alternating colors for chunks */
.chunk:nth-child(3n+1) {
  background-color: #e6f3ff;  /* Light blue for first chunk in group */
}

.chunk:nth-child(3n+2) {
  background-color: #fff0e6;  /* Light orange for second chunk in group */
}

.chunk:nth-child(3n+3) {
  background-color: #e6ffe6;  /* Light green for third chunk in group */
}

/* Hover effect for chunks */
.chunk:hover {
  transform: scale(1.02);      /* Slight scale effect on hover */
  cursor: pointer;            /* Change cursor to indicate interactivity */
}

.chunk:nth-child(3n+1):hover {
    background-color: #cce0ff;  /* Darker blue on hover */
}

.chunk:nth-child(3n+2):hover {
    background-color: #ffdbcc;  /* Darker orange on hover */
}

.chunk:nth-child(3n+3):hover {
    background-color: #ccffcc;  /* Darker green on hover */
}

/* Style for overlap underline */
.chunk:nth-child(3n+1) .overlap-underline {
    border-bottom: 5px solid #ffdbcc;  /* Orange for next chunk */
}
.chunk:nth-child(3n+2) .overlap-underline {
    border-bottom: 5px solid #ccffcc;  /* Green for next chunk */
}
.chunk:nth-child(3n+3) .overlap-underline {
    border-bottom: 5px solid #cce0ff;  /* Blue for next chunk */
}
.chunk:nth-child(3n+1) .overlap-underline-highlighted {
    border-bottom: 5px solid #ffb399;  /* Darker orange for next chunk */
}
.chunk:nth-child(3n+2) .overlap-underline-highlighted {
    border-bottom: 5px solid #99ff99;  /* Darker green for next chunk */
}
.chunk:nth-child(3n+3) .overlap-underline-highlighted {
    border-bottom: 5px solid #99ccff;  /* Darker blue for next chunk */
}

.chunk-line-break::after {
  content: "↵";  /* Unicode arrow symbol for line break */
  color: #999;
  vertical-align: top;
}
</style>
"""

chunk_container_scripts = """
<script>
// Add event listeners to all chunks
const doc = window.parent.document;
// get the notification div
let notification = doc.querySelector('.notification');

if (!notification) {
    notification = doc.createElement('div');
    notification.className = 'notification';
    doc.body.appendChild(notification);
}

const chunks = document.querySelectorAll('.chunk');
console.log('Chunks:', chunks);

let hoverTimeout;

function showMetadata(chunk) {
    let metadata = chunk.getAttribute('data-metadata');
    if (!metadata) {
        metadata = "{}";
    }
    const index = chunk.getAttribute('data-index');
    console.log('Chunk Metadata:', metadata);

    const parsedMetadata = JSON.parse(metadata);
    const formattedMetadata = JSON.stringify(parsedMetadata, null, 2);
    notification.innerHTML = `
        <button class="close-button" onclick="this.parentElement.style.display='none'">×</button>
        <h3 style="margin: 0 0 10px 0; color: #333;">Chunk ${index} Metadata</h3>
        <pre>${formattedMetadata}</pre>
    `;
    notification.style.display = 'block';
}

// Add hover effect for previous chunk's underline
function highlightPreviousUnderline(chunk) {
    const index = parseInt(chunk.getAttribute('data-index'));
    if (index > 0) {
        const previousChunk = document.querySelector(`.chunk[data-index="${index - 1}"]`);
        if (previousChunk) {
            const underline = previousChunk.querySelector('.overlap-underline');
            if (underline) {
                underline.classList.add('overlap-underline-highlighted');
            }
        }
    }
}

function resetPreviousUnderline(chunk) {
    const index = parseInt(chunk.getAttribute('data-index'));
    if (index > 0) {
        const previousChunk = document.querySelector(`.chunk[data-index="${index - 1}"]`);
        if (previousChunk) {
            const underline = previousChunk.querySelector('.overlap-underline');
            if (underline) {
                underline.classList.remove('overlap-underline-highlighted');
            }
        }
    }
}

chunks.forEach(chunk => {
    chunk.addEventListener('mouseenter', () => {
        hoverTimeout = setTimeout(() => {
            showMetadata(chunk);
            highlightPreviousUnderline(chunk);
        }, 200);
    });
    chunk.addEventListener('mouseleave', () => {
        clearTimeout(hoverTimeout);
        resetPreviousUnderline(chunk);
    });
    chunk.addEventListener('click', () => showMetadata(chunk));
});
</script>
"""


# Main Streamlit app
def main():
    # Create a mapping of splitter names to indices
    splitter_name_to_index = {splitter["name"]: i for i, splitter in enumerate(SPLITTERS)}
    
    # Handle query parameters
    if "splitter" in st.query_params:
        splitter_name = st.query_params["splitter"]
        splitter_index = splitter_name_to_index.get(splitter_name, 0)
    else:
        splitter_index = 0

    # Create a sidebar for splitter selection
    st.sidebar.title("Splitter Selection")

    # Create a vertical column for splitter selection
    selected_splitter_index = st.sidebar.radio(
        "Choose a text splitter:", 
        range(len(SPLITTERS)), 
        format_func=lambda i: SPLITTERS[i]["name"],
        index=splitter_index,
        key="selected_splitter_index",
        on_change=lambda: st.query_params.update({"splitter": SPLITTERS[st.session_state.selected_splitter_index]["name"]})
    )

    # Update query params when splitter changes
    st.query_params["splitter"] = SPLITTERS[selected_splitter_index]["name"]

    selected_splitter = SPLITTERS[selected_splitter_index]
    selected_splitter_name = selected_splitter["name"]

    # Add a separator
    st.sidebar.divider()

    # Add text selection section
    st.sidebar.title("Text Selection")

    # Create a dictionary of text examples
    text_examples = {
        "🎯 Showcase": selected_splitter.get("showcase_text") or TEXT_PLAIN,
        "📝 Plain Text": TEXT_PLAIN,
        "🐍 Python Code": TEXT_PYTHON,
        "🖥️ JavaScript Code": TEXT_JS,
        "📄 Markdown": TEXT_MARKDOWN,
        "🌐 HTML": TEXT_HTML,
    }

    # Create a vertical column for text selection
    selected_text_name = st.sidebar.radio("Choose example text:", list(text_examples.keys()))

    # Initialize session state for text if not exists
    if "input_text" not in st.session_state and "showcase_text" in selected_splitter:
        st.session_state.input_text = selected_splitter["showcase_text"]

    # Update text only if it's not the showcase option or if it hasn't been modified
    if selected_text_name != "🎯 Showcase" or not st.session_state.get("text_modified", False):
        st.session_state.input_text = text_examples[selected_text_name]

    # Add token counting section
    st.sidebar.divider()
    st.sidebar.title("Token Counting")

    # Add a checkbox to enable/disable token counting
    show_token_count = st.sidebar.checkbox("Calculate token counts", value=True)

    if show_token_count:
        # Get current text
        current_text = st.session_state.input_text
        
        # Create columns for token counts
        cols = st.sidebar.columns(len(TOKENIZERS) + 1)  # +1 for char count
        
        # Show character count first
        with cols[0]:
            st.markdown("**Characters**")
            st.markdown(f"Count: {len(current_text)}")
        
        # Show token counts for each tokenizer
        for i, (tokenizer_name, tokenizer) in enumerate(TOKENIZERS.items(), start=1):
            with cols[i]:
                st.markdown(f"**{tokenizer_name}**")
                tokens = tokenizer.encode(current_text)
                st.markdown(f"Tokens: {len(tokens)}")
    if link_to_docs := selected_splitter.get("link"):
        st.subheader(f"[{selected_splitter_name}]({link_to_docs})")
    else:
        st.subheader(selected_splitter_name)
    with st.expander(f"{selected_splitter['class_'].__name__} **Documentation** and its baseclasses", expanded=False):
        st.help(selected_splitter["class_"])
        # also baseclasses
        for base in selected_splitter["class_"].__bases__:
            if base is not object:  # Skip builtins.object()
                st.help(base)
    # Get custom parameters from user
    with st.expander("Splitter Parameters", expanded=True):
        custom_params = adapt_langchain_splitter_to_streamlit(selected_splitter)

    # Text input area
    st.subheader("Input Text")
    
    input_text = st.text_area(
        "Enter text to split:",
        value=st.session_state.input_text,
        height=200,
        key="text_area",
        on_change=lambda: setattr(st.session_state, "text_modified", True),
    )

    # Update session state with current text
    st.session_state.input_text = input_text

    def perform_split():
        if not input_text:
            st.warning("Please enter some text to split.")
            return

        try:
            # Create the splitter instance with custom parameters
            _kwargs = custom_params.pop("kwargs", None)
            if _kwargs is not None:
                custom_params.update(_kwargs)
            pprint(custom_params)
            splitter = selected_splitter["class_"](**custom_params)
            
            # Start timing
            start_time = time.perf_counter()
            
            if hasattr(splitter, "create_documents"):
                chunks = splitter.create_documents([input_text])
                # chunks = [chunk for chunk in chunks if (chunk.metadata.get("start_index") != -1)]
            elif issubclass(type(splitter), chonkie.BaseChunker):
                _chunks: list[chonkie.Chunk] = splitter.chunk(input_text)
                chunks = []
                for _chunk in _chunks:
                    metadata = _chunk.to_dict()
                    metadata.pop("text")
                    chunks.append(Document(page_content=_chunk.text, metadata=metadata))
            else:
                chunks = splitter.split_text(input_text)
            
            # End timing
            end_time = time.perf_counter()
            split_duration = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Get overlaps between chunks
            overlaps = identify_overlaps(
                chunks, chunk_size=custom_params.get("chunk_size"), chunk_overlap=custom_params.get("chunk_overlap")
            )

            # Add token counts to each chunk's metadata if enabled
            if show_token_count:
                for chunk in chunks:
                    if isinstance(chunk, Document):
                        text = chunk.page_content
                    else:
                        text = chunk
                    
                    # Add character count
                    chunk.metadata["char_count"] = len(text)
                    
                    # Add token counts for all tokenizers
                    for tokenizer_name, tokenizer in TOKENIZERS.items():
                        chunk.metadata[f"tokens_{tokenizer_name}"] = len(tokenizer.encode(text))

            # Display results
            st.subheader(f"Split Results (took {split_duration:.1f} ms)")
            content = chunk_container_styles
            content += "<div class='chunk-container'>"

            def rreplace(s, old, new):
                return new.join(s.rsplit(old, 1))

            for i, chunk in enumerate(chunks):
                if isinstance(chunk, Document):
                    page_content = chunk.page_content
                    metadata = chunk.metadata
                    metadata["char_count"] = len(page_content)
                    metadata["text"] = page_content
                elif isinstance(chunk, str):
                    page_content = chunk
                    metadata = {"char_count": len(chunk)}
                else:
                    raise ValueError(f"Unknown chunk type: {type(chunk)}")

                overlap_prefix = None
                overlap_suffix = None

                if i - 1 in overlaps:  # this is right chunk, do not show the overlap as it is already in the left chunk
                    _, overlap_prefix_index = overlaps[i - 1]
                    overlap_prefix = page_content[:overlap_prefix_index]
                    metadata["overlap_prefix"] = overlap_prefix
                    page_content = page_content[overlap_prefix_index:]

                if i in overlaps:  # this is left chunk, underline the overlap
                    overlap_suffix_index, _ = overlaps[i]
                    overlap_suffix = page_content[overlap_suffix_index:]
                    metadata["overlap_suffix"] = overlap_suffix
                    page_content = page_content[:overlap_suffix_index] + "[OVERLAP-SUFFIX]"

                escaped = escape(page_content)

                if overlap_suffix is not None:
                    escaped = rreplace(
                        escaped, "[OVERLAP-SUFFIX]", f'<span class="overlap-underline">{overlap_suffix}</span>'
                    )

                escaped = escaped.replace("\n", "<span class='chunk-line-break'></span>\n")
                metadata = escape(json.dumps(metadata, default=str))
                content += f"<span class='chunk' data-index='{i}' data-metadata='{metadata}'>{escaped}</span>"

            content += "</div>"
            content += chunk_container_scripts

            html(content, scrolling=True, height=1000)
        except Exception as e:
            st.error(f"Error during text splitting: {str(e)}")
            st.error("Please check your input text and splitter parameters.")
            raise e

    # Perform split automatically when text or parameters change
    perform_split()


if __name__ == "__main__":
    main()
