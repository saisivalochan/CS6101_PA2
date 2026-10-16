"""Network-free numerical integration tests; run by cloud bootstrap after install."""
import importlib.util
import json
import pathlib
import tempfile
import types
import unittest
from unittest.mock import patch

AVAILABLE=all(importlib.util.find_spec(name) is not None for name in ("numpy","faiss","sklearn","datasets","reportlab","matplotlib","threadpoolctl"))


@unittest.skipUnless(AVAILABLE,"numeric/report dependencies are not installed")
class IntegrationTests(unittest.TestCase):
    def test_complete_miniature_workflow_report_audit_and_packaging(self):
        import numpy as np
        from threadpoolctl import threadpool_limits
        from pa2 import data
        from pa2.audit import audit
        from pa2.experiments import build,truth,evaluate,scaling_summary
        from pa2.report import generate,package
        from pa2.util import ROOT,load_config,write_json
        import zipfile
        rng=np.random.default_rng(81)
        with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=1):
            root=pathlib.Path(tmp)
            config=load_config(ROOT/"configs/main.json")
            for key in ("data_dir","index_dir","results_dir"):config[key]=str(root/key)
            config["hnsw"].update({"M":8,"efConstruction":30})
            config["vamana"].update({"R":8,"L_build":20})
            config["study"]=[{"R":8,"L_build":20,"alpha":1.2},{"R":4,"L_build":20,"alpha":1.2},{"R":8,"L_build":20,"alpha":1.0}]
            config["scaling_sizes"]=[100,150,200]
            # All fixtures are temporary and explicitly synthetic; never counted
            # as the real full-data assignment experiments.
            for dataset in ["sift100k",*config["full_datasets"]]:
                directory=data.paths(config,dataset);directory.mkdir(parents=True)
                vectors=rng.integers(0,256,size=(200,128)).astype(np.float32)
                queries=rng.integers(0,256,size=(12,128)).astype(np.float32)
                data._save_array(directory/"corpus.npy",vectors);data._save_array(directory/"queries.npy",queries)
                artifacts=["corpus.npy","queries.npy"]
                if dataset=="sift1m":
                    neighbors=np.argsort(np.sum((vectors[None,:,:]-queries[:,None,:])**2,axis=2),axis=1)[:,:10]
                    data._save_array(directory/"supplied_truth.npy",neighbors);artifacts.append("supplied_truth.npy")
                data._finish(directory,{"dataset":dataset,"corpus_size":200,"query_count":12,"original_dimension":128,
                    "search_dimension":128,"metric":"L2","normalized":False,"raw_corpus_bytes":200*128*4,
                    "source":"SYNTHETIC temporary integration fixture, not assignment measurements"},artifacts)
                build(config,dataset,"exact",{});truth(config,dataset)
                if dataset in config["full_datasets"]:
                    evaluate(config,dataset,"exact",{},"exact")
                    build(config,dataset,"hnsw",config["hnsw"])
                    for setting in config["hnsw"]["efSearch"]:evaluate(config,dataset,"hnsw",config["hnsw"],setting)
                graph_params=config["study"] if dataset=="sift100k" else [config["vamana"]]
                for params in graph_params:
                    build(config,dataset,"vamana",params)
                    for setting in config["vamana"]["L_search"]:evaluate(config,dataset,"vamana",params,setting)
            for size in config["scaling_sizes"]:
                name=f"sift{size}"
                data.prepare_subset(config,size,name)
                build(config,name,"vamana",config["vamana"])
            scaling_summary(config)
            details=root/"details.json";transcript=root/"chat.txt"
            transcript.write_text("Synthetic test-only transcript fixture. This is not an actual chat or assignment submission. "*2)
            write_json(details,{"roll_numbers":["TESTONLY"],"llm_documentation_complete":True,"llm_chats":[{
                "description":"Synthetic integration-test fixture only","share_url":"https://chatgpt.com/share/synthetic-test-fixture",
                "transcript_path":"chat.txt"}]})
            summary,_=audit(config,details)
            self.assertTrue(summary["complete"],summary["missing_or_invalid"])
            self.assertEqual(summary["available_operating_points"],48)
            pdf=generate(config,details)
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))
            self.assertTrue(audit(config,details,include_report=True)[0]["complete"])
            archive=package(config,details,root/"submission")
            with zipfile.ZipFile(archive) as bundle:
                self.assertIn("results/report.pdf",bundle.namelist())
                self.assertFalse(any(name.startswith(("data/","indices/")) for name in bundle.namelist()))
            transcript.write_text("Changed transcript invalidates the generated report. "*3)
            self.assertFalse(audit(config,details,include_report=True)[0]["complete"])

    def test_faiss_vamana_truth_timing_and_plot_round_trip(self):
        import numpy as np
        from threadpoolctl import threadpool_limits
        from pa2 import data
        from pa2.experiments import build,truth,evaluate
        from pa2.report import _plots
        from pa2.util import ROOT,load_config
        rng=np.random.default_rng(10)
        with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=1):
            config=load_config(ROOT/"configs/main.json")
            for key in ("data_dir","index_dir","results_dir"):config[key]=str(pathlib.Path(tmp)/key)
            config["hnsw"].update({"M":8,"efConstruction":30})
            config["vamana"].update({"R":8,"L_build":20})
            for dataset in ("sift1m","sift100k"):
                root=pathlib.Path(config["data_dir"])/dataset;root.mkdir(parents=True)
                base=rng.integers(0,256,size=(200,128)).astype(np.float32)
                queries=rng.integers(0,256,size=(12,128)).astype(np.float32)
                data._save_array(root/"corpus.npy",base);data._save_array(root/"queries.npy",queries)
                artifacts=["corpus.npy","queries.npy"]
                if dataset=="sift1m":
                    supplied=np.argsort(np.sum((base[None,:,:]-queries[:,None,:])**2,axis=2),axis=1)[:,:10]
                    data._save_array(root/"supplied_truth.npy",supplied);artifacts.append("supplied_truth.npy")
                data._finish(root,{"dataset":dataset,"corpus_size":200,"query_count":12,"original_dimension":128,
                    "search_dimension":128,"metric":"L2","normalized":False,"raw_corpus_bytes":200*128*4,
                    "source":"synthetic unit-test fixture, not assignment evidence"},artifacts)
                build(config,dataset,"exact",{});truth(config,dataset)
                methods=[("vamana",config["vamana"],100)]
                if dataset=="sift1m":methods += [("exact",{},"exact"),("hnsw",config["hnsw"],100)]
                for method,params,setting in methods:
                    first=build(config,dataset,method,params)
                    self.assertEqual(first,build(config,dataset,method,params))
                    result=evaluate(config,dataset,method,params,setting)
                    self.assertEqual(result,evaluate(config,dataset,method,params,setting))
                    self.assertTrue(0<=result["recall_10"]<=1)
                    self.assertGreater(result["mean_latency_ms"],0)
                    self.assertEqual(result["evaluation_queries"],12)
            point_paths=list((pathlib.Path(config["results_dir"])/"points").glob("*.json"))
            evidence={"points":[json.loads(p.read_text()) for p in point_paths]}
            figures=_plots(config,evidence)
            self.assertEqual(len(figures),4)
            self.assertTrue(all(p.stat().st_size>1000 for p in figures))
            self.assertTrue((pathlib.Path(config["results_dir"])/"sift100k_trace.json").exists())

    def test_ir_pca_is_corpus_only_shared_and_normalized(self):
        import numpy as np
        from threadpoolctl import threadpool_limits
        from pa2 import data
        from pa2.util import ROOT,load_config
        rng=np.random.default_rng(14)
        with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=1):
            config=load_config(ROOT/"configs/main.json")
            config["data_dir"]=tmp;config["pca_sample"]=150
            base=rng.normal(size=(200,1024)).astype(np.float32)
            queries=rng.normal(size=(8,1024)).astype(np.float32)
            root=pathlib.Path(tmp)
            for name,values in (("corpus",base),("queries",queries)):
                values.tofile(root/f"{name}.f32")
                (root/f"{name}.jsonl").write_text("".join(json.dumps(str(i))+"\n" for i in range(len(values))))
            def stream(config,dataset,collection,revision):
                return root/f"{collection}.f32",root/f"{collection}.jsonl",len(base if collection=="corpus" else queries)
            with patch("huggingface_hub.HfApi",return_value=types.SimpleNamespace(dataset_info=lambda _:types.SimpleNamespace(sha="synthetic-test-revision"))),patch("pa2.data._stream_raw",side_effect=stream):
                info=data.prepare_ir(config,"quora")
                prepared=data.paths(config,"quora")
                corpus1=np.load(prepared/"corpus.npy");query1=np.load(prepared/"queries.npy")
                with np.load(prepared/"pca.npz") as pca:
                    mean=pca["mean"].copy();components=pca["components"].copy();sample=pca["sample_rows"].copy()
                self.assertEqual(len(sample),150)
                self.assertTrue(np.allclose(mean,base[sample].mean(axis=0),atol=1e-6))
                expected=(queries-mean)@components.T;expected=expected.astype(np.float32)
                expected/=np.linalg.norm(expected,axis=1,keepdims=True)
                self.assertTrue(np.allclose(query1,expected,atol=1e-5))
                self.assertTrue(np.allclose(np.linalg.norm(corpus1,axis=1),1,atol=1e-6))
                queries[:]=rng.normal(100,10,size=queries.shape);queries.tofile(root/"queries.f32")
                data.prepare_ir(config,"quora",force=True)
                self.assertTrue(np.allclose(corpus1,np.load(prepared/"corpus.npy"),atol=1e-6))
                self.assertFalse(np.allclose(query1,np.load(prepared/"queries.npy")))
                self.assertEqual(info["pca"]["fitting_source"],"corpus only")


if __name__=="__main__":unittest.main()
