# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Email (.eml) parser

import os
from typing import Dict, Any, List


class EmailParser:
    """Parser for RFC 822/EML email files"""

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse an EML file into plain text
        
        Args:
            file_path: Path to the .eml file
            
        Returns:
            A list with a single dict containing extracted text
        """
        from email import policy
        from email.parser import BytesParser

        with open(file_path, "rb") as f:
            data = f.read()

        msg = BytesParser(policy=policy.default).parsebytes(data)

        # Prefer text/plain parts
        texts: List[str] = []
        html_texts: List[str] = []

        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = part.get_content_disposition()
                # skip attachments
                if disp == "attachment":
                    continue
                if ctype == "text/plain":
                    try:
                        texts.append(part.get_content())
                    except:  # noqa: E722
                        payload = part.get_payload(decode=True) or b""
                        texts.append(payload.decode(part.get_content_charset() or "utf-8", errors="ignore"))
                elif ctype == "text/html":
                    try:
                        html_texts.append(part.get_content())
                    except:  # noqa: E722
                        payload = part.get_payload(decode=True) or b""
                        html_texts.append(payload.decode(part.get_content_charset() or "utf-8", errors="ignore"))
        else:
            ctype = msg.get_content_type()
            if ctype == "text/plain":
                texts.append(msg.get_content())
            elif ctype == "text/html":
                html_texts.append(msg.get_content())

        if not texts and html_texts:
            # Fallback: extract text from HTML using bs4 if available, otherwise strip tags
            combined_html = "\n\n".join(html_texts)
            try:
                import bs4
                soup = bs4.BeautifulSoup(combined_html, "html.parser")
                texts.append(soup.get_text(separator="\n").strip())
            except ImportError:
                import re
                texts.append(re.sub(r"<[^>]+>", "", combined_html))

        full_text = "\n\n".join(t.strip() for t in texts if t and t.strip())
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
