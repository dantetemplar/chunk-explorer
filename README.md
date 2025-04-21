# Chunk Explorer

A powerful tool for exploring and visualizing different text chunking strategies, designed to help you understand how various text splitters work with different types of content.

## 🚀 [Try it here](https://chunkers.streamlit.app/)


https://github.com/user-attachments/assets/3aabf071-15f6-4302-9678-a76b6c766123


## Features

The Chunk Explorer provides:

- **Multiple Chunking Strategies**: Choose from various text splitters to see how they handle different types of content
- **Interactive Visualization**: See chunks with alternating colors and hover effects for better visualization
- **Metadata Display**: View detailed metadata for each chunk on hover or click
- **Overlap Visualization**: See how chunks overlap with highlighted underlines
- **Token Counting**: Calculate token counts using different tokenizers
- **Multiple Text Types**: Test with various text formats:
  - Plain text
  - Python code
  - JavaScript code
  - Markdown
  - HTML

## Example

Here's an example of how the Chunk Explorer visualizes text chunks:

![image](https://github.com/user-attachments/assets/a5cf9b3e-eda1-4670-a692-8f29bacd07fa)

Each chunk is color-coded and interactive:
- Hover over chunks to see metadata
- Click on chunks for persistent metadata display
- Overlapping regions are highlighted with colored underlines
- Line breaks are clearly marked with arrow symbols

## Development

1. Install dependencies using `uv`:

    ```bash
    uv sync
    ```

2. Run the Streamlit app:

    ```bash
    uv run streamlit run app.py
    ```

3. Use the sidebar to:
   - Select different text splitters
   - Choose example text types
   - Configure splitter parameters
   - Enable/disable token counting

### Adding Dependencies
```bash
uv add <package-name>
```

## Project Structure

- `app.py`: Main Streamlit application
- `strategies.py`: Contains text splitting strategies and tokenizers
- `texts.py`: Sample text content for different formats
