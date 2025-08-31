import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from synthetic_data_kit.parsers.epub_parser import EPUBParser
from synthetic_data_kit.parsers.email_parser import EmailParser


class TestEPUBParser:
    """Tests for EPUB parser"""

    def test_epub_parser_initialization(self):
        parser = EPUBParser()
        assert parser is not None

    @patch("builtins.__import__")
    def test_epub_parse_success(self, mock_import):
        """Test parsing EPUB with multiple HTML documents inside"""
        # Build mock ebooklib with epub.read_epub and ITEM_DOCUMENT
        mock_ebooklib = MagicMock()
        mock_epub = MagicMock()

        class ItemWithMethod:
            def get_content(self):
                return b"<html><body><p>First</p></body></html>"

        class ItemWithAttr:
            def __init__(self):
                self.content = b"<html><body><p>Second</p></body></html>"

        mock_book = MagicMock()
        mock_book.get_items_of_type.return_value = [ItemWithMethod(), ItemWithAttr()]
        mock_ebooklib.ITEM_DOCUMENT = 1
        mock_epub.read_epub.return_value = mock_book
        mock_ebooklib.epub = mock_epub

        # Mock bs4.BeautifulSoup to extract simple text
        mock_bs4 = MagicMock()

        def bs_factory(html, parser):
            soup = MagicMock()
            if "First" in html:
                soup.get_text.return_value = "First"
            else:
                soup.get_text.return_value = "Second"
            return soup

        mock_bs4.BeautifulSoup = MagicMock(side_effect=bs_factory)

        def import_side_effect(name, *args, **kwargs):
            if name == "ebooklib":
                return mock_ebooklib
            if name == "bs4":
                return mock_bs4
            return MagicMock()

        mock_import.side_effect = import_side_effect

        parser = EPUBParser()
        result = parser.parse("/fake/path/book.epub")

        assert result == [{"text": "First\n\nSecond"}]

    @patch("builtins.__import__")
    def test_epub_parse_import_error(self, mock_import):
        mock_import.side_effect = lambda name, *a, **k: (_ for _ in ()).throw(ImportError("No module named 'ebooklib'")) if name == "ebooklib" else MagicMock()
        parser = EPUBParser()
        with pytest.raises(ImportError, match="ebooklib is required"):
            parser.parse("/fake/path/book.epub")


class TestEmailParser:
    """Tests for Email (.eml) parser"""

    def test_email_parser_initialization(self):
        parser = EmailParser()
        assert parser is not None

    def test_email_parse_plain_text(self):
        # Simple singlepart plain text email
        eml = (
            "From: a@example.com\n"
            "To: b@example.com\n"
            "Subject: Test\n"
            "MIME-Version: 1.0\n"
            "Content-Type: text/plain; charset=utf-8\n\n"
            "Hello\nWorld"
        ).encode("utf-8")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as f:
            f.write(eml)
            path = f.name
        try:
            parser = EmailParser()
            result = parser.parse(path)
            assert result == [{"text": "Hello\nWorld"}]
        finally:
            os.unlink(path)

    def test_email_parse_multipart_html_fallback(self):
        # Multipart with only HTML content; should fallback via bs4
        boundary = "===============123456=="
        html_body = "<html><body><p>HTML Only</p></body></html>"
        eml = (
            f"From: a@example.com\n"
            f"To: b@example.com\n"
            f"Subject: Test HTML\n"
            f"MIME-Version: 1.0\n"
            f"Content-Type: multipart/alternative; boundary=\"{boundary}\"\n\n"
            f"--{boundary}\n"
            f"Content-Type: text/html; charset=utf-8\n\n"
            f"{html_body}\n"
            f"--{boundary}--\n"
        ).encode("utf-8")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as f:
            f.write(eml)
            path = f.name

        # Patch only bs4 import
        import builtins as _builtins
        real_import = _builtins.__import__

        mock_bs4 = MagicMock()
        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "HTML Only"
        mock_bs4.BeautifulSoup.return_value = mock_soup

        def import_side_effect(name, *args, **kwargs):
            if name == "bs4":
                return mock_bs4
            return real_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=import_side_effect):
                parser = EmailParser()
                result = parser.parse(path)
                assert result == [{"text": "HTML Only"}]
        finally:
            os.unlink(path)

    def test_email_parse_skip_attachments(self):
        # Multipart/mixed with attachment; parser should ignore attachment and keep text
        boundary = "===============654321=="
        eml = (
            f"From: a@example.com\n"
            f"To: b@example.com\n"
            f"Subject: With Attachment\n"
            f"MIME-Version: 1.0\n"
            f"Content-Type: multipart/mixed; boundary=\"{boundary}\"\n\n"
            f"--{boundary}\n"
            f"Content-Type: text/plain; charset=utf-8\n\n"
            f"Body text here.\n"
            f"--{boundary}\n"
            f"Content-Type: application/octet-stream\n"
            f"Content-Disposition: attachment; filename=\"file.bin\"\n\n"
            f"BINARYDATA\n"
            f"--{boundary}--\n"
        ).encode("utf-8")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as f:
            f.write(eml)
            path = f.name
        try:
            parser = EmailParser()
            result = parser.parse(path)
            assert result == [{"text": "Body text here."}]
        finally:
            os.unlink(path)
