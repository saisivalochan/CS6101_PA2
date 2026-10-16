import base64
import pathlib
import tempfile
import unittest

from pa2.report import transcript_attachments
from scripts.export_llm_chats import preserve_image_attachments


class ChatExportTests(unittest.TestCase):
    def test_inline_image_is_preserved_without_printing_binary_encoding(self):
        pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aU1sAAAAASUVORK5CYII="
        text = f'visible question; data:image/png;base64,{pixel}; visible answer'
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            exported = preserve_image_attachments(text, root)
            self.assertTrue(exported.startswith("visible question; "))
            self.assertTrue(exported.endswith("; visible answer"))
            self.assertNotIn(pixel, exported)
            images = transcript_attachments(exported, root)
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0].read_bytes(), base64.b64decode(pixel))
            self.assertEqual(preserve_image_attachments(text, root), exported)

    def test_changed_attachment_is_rejected_before_report_or_packaging(self):
        pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aU1sAAAAASUVORK5CYII="
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            exported = preserve_image_attachments(f'data:image/png;base64,{pixel}', root)
            image = transcript_attachments(exported, root)[0]
            image.write_bytes(b"changed image")
            with self.assertRaisesRegex(ValueError, "attachment checksum mismatch"):
                transcript_attachments(exported, root)


if __name__ == "__main__":
    unittest.main()
