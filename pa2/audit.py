"""Check evidence for the compulsory assignment, without importing numeric packages."""
import pathlib
from urllib.parse import urlparse

from . import data
from .util import fingerprint, index_id, read_json, sha256, write_json


def submission_details(path):
    path=pathlib.Path(path)
    problems=[]
    if not path.exists():
        return {},["Submission details file is absent (roll numbers, LLM documentation, chat links)"]
    value=read_json(path)
    rolls=value.get("roll_numbers",[])
    if not 1<=len(rolls)<=2 or any(not isinstance(x,str) or not x or not all(c.isalnum() or c=="-" for c in x) for x in rolls):
        problems.append("Provide one or two valid roll numbers")
    chats=value.get("llm_chats",[])
    if not chats or not value.get("llm_documentation_complete",False):
        problems.append("Full LLM documentation and confirmation that all used chats are included are required")
    for number,chat in enumerate(chats,1):
        url=urlparse(chat.get("share_url",""))
        if url.scheme!="https" or not url.netloc or url.hostname in ("example.com","example.org","localhost") or not url.path.strip("/"):
            problems.append(f"Chat {number}: provide a real HTTPS shareable chat URL")
        transcript=path.parent/chat.get("transcript_path","")
        if not transcript.is_file() or transcript.stat().st_size<50:
            problems.append(f"Chat {number}: full readable transcript file is missing")
        if not chat.get("description","").strip():
            problems.append(f"Chat {number}: describe its use in the assignment")
    return value,problems


def audit(config,details_path=None,include_report=False,verify_checksums=True):
    root=pathlib.Path(config["results_dir"])
    problems=[];checks=[];datasets={};builds={};points=[]
    for dataset in ["sift100k",*config["full_datasets"]]:
        try:
            datasets[dataset]=data.manifest(config,dataset,verify_checksums)
            checks.append(f"prepared dataset: {dataset}")
        except (OSError,ValueError,KeyError) as error:
            problems.append(f"{dataset} preparation: {error}")
    expected=[]
    for dataset in config["full_datasets"]:
        expected.append((dataset,"exact",{},["exact"]))
        expected.append((dataset,"hnsw",config["hnsw"],config["hnsw"]["efSearch"]))
        expected.append((dataset,"vamana",config["vamana"],config["vamana"]["L_search"]))
    for params in config["study"]:
        expected.append(("sift100k","vamana",params,config["vamana"]["L_search"]))
    hosts=set()
    for dataset,method,params,settings in expected:
        name=index_id(method,params);key=f"{dataset}__{name}"
        path=root/"builds"/f"{key}.json"
        try:
            build=read_json(path)
            if dataset in datasets and build["spec"]["dataset_fingerprint"]!=datasets[dataset]["fingerprint"]:
                raise ValueError("build uses different vectors")
            if build["build_seconds"]<=0 or build["build_process_peak_rss_bytes"]<=0 or build["index_bytes"]<=0:
                raise ValueError("missing positive build measurements")
            if method=="vamana" and (build["maximum_out_degree"]>params["R"] or not build.get("validation")):
                raise ValueError("degree validation missing/failed")
            extension="vamana" if method=="vamana" else "faiss"
            artifact=pathlib.Path(config["index_dir"])/dataset/f"{name}.{extension}"
            if not artifact.exists() or artifact.stat().st_size!=build["serialized_bytes"]:
                raise ValueError("persisted index missing/size mismatch")
            if verify_checksums and sha256(artifact)!=build["index_sha256"]:
                raise ValueError("persisted index checksum mismatch")
            builds[key]=build
        except (OSError,ValueError,KeyError) as error:
            problems.append(f"{key} build: {error}")
            build=None
        for setting in settings:
            try:
                point=read_json(root/"points"/f"{key}__{setting}.json")
                if build and point["index_signature"]!=build["signature"]:
                    raise ValueError("operating point uses another index")
                if dataset in datasets and point["evaluation_queries"]!=datasets[dataset]["query_count"]:
                    raise ValueError("recall was not evaluated on all queries")
                if not 0<=point["recall_1"]<=1 or not 0<=point["recall_10"]<=1:
                    raise ValueError("recall outside [0,1]")
                if point["mean_latency_ms"]<=0 or point["qps"]<=0:
                    raise ValueError("invalid timing")
                if point["policy"]["threads"]!=1 or point["policy"]["query_mode"]!="individual":
                    raise ValueError("inconsistent query mode/threads")
                timing=read_json(root/"timing"/f"{key}__{setting}.json")
                if len(timing["runs"])<3 or point["timing_query_count"]<min(1000,point["evaluation_queries"]):
                    raise ValueError("too few timing repetitions/queries")
                if method=="vamana" and (point["vertices_expanded_per_query"]<=0 or point["distance_evaluations_per_query"]<point["vertices_expanded_per_query"]):
                    raise ValueError("invalid search counters")
                host=point["machine"].get("host")
                if host:hosts.add(host)
                points.append(point)
            except (OSError,ValueError,KeyError) as error:
                problems.append(f"{key} setting {setting}: {error}")
    if len(hosts)>1:
        problems.append("Query timings were recorded on different hosts; rerun comparisons on the same VM")
    for dataset in ["sift100k",*config["full_datasets"]]:
        try:
            info=read_json(root/"truth"/f"{dataset}.json")
            if dataset in datasets and info["dataset_fingerprint"]!=datasets[dataset]["fingerprint"]:
                raise ValueError("ground truth uses different prepared vectors")
            if dataset=="sift1m" and not info["supplied_distances_verified"]:
                raise ValueError("supplied SIFT truth agreement not verified")
        except (OSError,ValueError,KeyError) as error:
            problems.append(f"{dataset} exact-truth evidence: {error}")
    for name in ["sift100k_trace.json","scaling.json"]:
        if not (root/name).exists():problems.append(f"Missing {name}")
    if (root/"scaling.json").exists():
        try:
            measured=read_json(root/"scaling.json")["measurements"]
            if {m["points"] for m in measured}!=set(config["scaling_sizes"]):
                raise ValueError("required subset sizes missing")
            for measurement in measured:
                size=measurement["points"]
                dataset="sift100k" if size==100000 else f"sift{size}"
                params=config["vamana"]
                build=read_json(root/"builds"/f"{dataset}__{index_id('vamana',params)}.json")
                if measurement["signature"]!=build["signature"] or measurement["build_seconds"]!=build["build_seconds"] or measurement["build_seconds"]<=0:
                    raise ValueError("scaling/build measurements disagree")
        except (OSError,ValueError,KeyError) as error:
            problems.append(f"Scaling evidence: {error}")
    if details_path is not None:
        _,detail_problems=submission_details(details_path);problems.extend(detail_problems)
    if include_report:
        try:
            report=read_json(root/"report_manifest.json")
            if report["draft"]:raise ValueError("report is a draft")
            if sha256(root/"report.pdf")!=report["pdf_sha256"]:
                raise ValueError("report PDF checksum mismatch")
            if details_path and sha256(details_path)!=report["details_sha256"]:
                raise ValueError("submission details changed after report generation")
            if details_path:
                for name,digest in report["transcript_sha256"].items():
                    if sha256(pathlib.Path(details_path).parent/name)!=digest:
                        raise ValueError("chat transcript changed after report generation")
                for name,digest in report.get("attachment_sha256",{}).items():
                    if sha256(pathlib.Path(details_path).parent/name)!=digest:
                        raise ValueError("chat image attachment changed after report generation")
            expected_figures={f"{dataset}_{suffix}.png" for dataset in config["full_datasets"] for suffix in ("hnsw","comparison")}
            expected_figures.update(["sift100k_search_list.png","sift100k_latency.png"])
            if set(report["figure_sha256"])!=expected_figures:
                raise ValueError("required figures missing")
            for name,digest in report["figure_sha256"].items():
                if sha256(root/"figures"/name)!=digest:
                    raise ValueError("figure changed after report generation")
            evidence=fingerprint({"datasets":datasets,"builds":builds,"points":points})
            if evidence!=report["evidence_fingerprint"]:
                raise ValueError("experiment evidence changed after report generation")
        except (OSError,ValueError,KeyError) as error:
            problems.append(f"Final report: {error}")
    summary={"complete":not problems,"checks":checks,"missing_or_invalid":problems,
             "available_operating_points":len(points),"available_builds":len(builds)}
    write_json(root/"audit.json",summary)
    return summary,{"datasets":datasets,"builds":builds,"points":points}
