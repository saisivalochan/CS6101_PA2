import array
import pathlib
import tempfile
import unittest

from pa2.core import Vamana
from scripts.finalize_local import HEADER, archive_file, normalized_index_sha256


class FinalizationTests(unittest.TestCase):
    def test_graph_comparison_ignores_elapsed_time_but_detects_edge_changes(self):
        points = array.array("f", range(40))
        with tempfile.TemporaryDirectory() as tmp, Vamana(points, 3, 2) as index:
            root = pathlib.Path(tmp)
            original, modified = root / "original", root / "modified"
            index.build(8, 1.2, 42, progress=lambda *args: False)
            index.save(original)
            data = bytearray(original.read_bytes())
            fields = HEADER.unpack(data[:HEADER.size])
            data[:HEADER.size] = HEADER.pack(*fields[:-1], fields[-1] + 100)
            modified.write_bytes(data)
            self.assertEqual(normalized_index_sha256(original), normalized_index_sha256(modified))
            data[HEADER.size + 20 * 4] ^= 1  # First adjacency ID after all degree records.
            modified.write_bytes(data)
            self.assertNotEqual(normalized_index_sha256(original), normalized_index_sha256(modified))

    def test_conflicting_archive_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source, destination = root / "source", root / "archived"
            source.write_bytes(b"original evidence")
            archive_file(source, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            source.write_bytes(b"changed evidence")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                archive_file(source, destination)
            self.assertEqual(destination.read_bytes(), b"original evidence")
            self.assertEqual(source.read_bytes(), b"changed evidence")


if __name__ == "__main__":
    unittest.main()
