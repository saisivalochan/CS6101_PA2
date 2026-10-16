import json
import pathlib
import tempfile
import unittest

from pa2.audit import audit,submission_details
from pa2.data import _finish,manifest
from pa2.experiments import recall
from pa2.util import ROOT,load_config,write_json


class WorkflowTests(unittest.TestCase):
    def test_recall_uses_sets_and_distinguishes_top1_from_top10(self):
        exact=[list(range(10)),list(range(10,20))]
        result=[[1,1,2,3,4,5,6,7,8,9],list(range(10,20))]
        quality=recall(result,exact)
        self.assertEqual(quality["recall_1"],.5)
        self.assertAlmostEqual(quality["recall_10"],.95)
        with self.assertRaises(ValueError):recall([],[])

    def test_manifest_rejects_same_size_content_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=pathlib.Path(tmp)/"toy";directory.mkdir()
            path=directory/"corpus.npy";path.write_bytes(b"test")
            _finish(directory,{"dataset":"toy"},["corpus.npy"])
            self.assertEqual(manifest({"data_dir":tmp},"toy",True)["dataset"],"toy")
            path.write_bytes(b"oops")
            with self.assertRaisesRegex(ValueError,"checksum"):
                manifest({"data_dir":tmp},"toy",True)

    def test_empty_results_cannot_pass_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=load_config(ROOT/"configs/main.json")
            for key in ("data_dir","index_dir","results_dir"):config[key]=str(pathlib.Path(tmp)/key)
            result,evidence=audit(config)
            self.assertFalse(result["complete"])
            self.assertEqual(result["available_operating_points"],0)
            self.assertGreater(len(result["missing_or_invalid"]),40)

    def test_missing_disclosure_and_fake_url_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);path=root/"details.json"
            write_json(path,{"roll_numbers":[],"llm_chats":[]})
            self.assertGreaterEqual(len(submission_details(path)[1]),2)
            (root/"chat.txt").write_text("A synthetic transcript fixture used only to test packaging validation. "*3)
            write_json(path,{"roll_numbers":["AB123"],"llm_documentation_complete":True,"llm_chats":[{
                "share_url":"https://example.com/fake","transcript_path":"chat.txt","description":"fixture"}]})
            self.assertTrue(any("real HTTPS" in x for x in submission_details(path)[1]))

    def test_final_report_refuses_incomplete_evidence_before_loading_pdf_tools(self):
        from pa2.report import generate
        with tempfile.TemporaryDirectory() as tmp:
            config=load_config(ROOT/"configs/main.json")
            for key in ("data_dir","index_dir","results_dir"):config[key]=str(pathlib.Path(tmp)/key)
            with self.assertRaisesRegex(ValueError,"Cannot generate a final report"):
                generate(config,pathlib.Path(tmp)/"missing.json")
            self.assertFalse((pathlib.Path(config["results_dir"])/"report.pdf").exists())


if __name__=="__main__":unittest.main()
