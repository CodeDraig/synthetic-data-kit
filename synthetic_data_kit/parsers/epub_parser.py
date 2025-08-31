# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# EPUB parser

import os
from typing import Dict, Any, List


class EPUBParser:
    """Parser for EPUB files"""

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse an EPUB file into plain text
        
        Args:
            file_path: Path to the EPUB file
            
        Returns:
            A list with a single dict containing extracted text from the document
        """
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError:
            raise ImportError(
                "ebooklib is required for EPUB parsing. Install it with: pip install ebooklib"
            )

        try:
            import bs4
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is required for EPUB parsing. Install it with: pip install beautifulsoup4"
            )

        book = epub.read_epub(file_path)

        texts: List[str] = []
        # Iterate over all document items (HTML files inside the EPUB)
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            content = item.get_content() if hasattr(item, "get_content") else getattr(item, "content", b"")
            if isinstance(content, (bytes, bytearray)):
                html = content.decode("utf-8", errors="ignore")
            else:
                html = str(content)
            soup = bs4.BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n").strip()
            if text:
                texts.append(text)

        full_text = "\n\n".join(t for t in texts if t)
        return [{"text": full_text}]

    def save(self, content: str, output_path: str) -> None:
        """Save the extracted text to a file
        
        Args:
            content: Extracted text content
            output_path: Path to save the text
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
