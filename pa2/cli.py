"""Command line entry point for the complete compulsory experiment workflow."""
import argparse
import array
import importlib.util
import json
import pathlib
import random
import subprocess
import sys
import time
import traceback

from .util import ROOT,configure_threads,load_config,machine_info,peak_rss_bytes,read_json,sha256,write_json


def _dependencies():
    missing=[name for name in ("numpy","faiss","sklearn","datasets","matplotlib","reportlab","threadpoolctl") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Missing dependencies: "+", ".join(missing)+". On Ubuntu run bash cloud/bootstrap.sh; see README.md. No experiment results have been created.")


def _prepare(config,dataset,archive=None,force=False):
    from .data import prepare_ir,prepare_sift,prepare_subset
    if dataset in ("all","sift1m"):
        prepare_sift(config,archive,force)
    if dataset in ("all","sift100k"):
        prepare_subset(config,100000,force=force)
    for name in ("quora","nq"):
        if dataset in ("all",name):prepare_ir(config,name,force)


def _worker(config,args):
    _dependencies()
    from threadpoolctl import threadpool_limits
    from .experiments import build,evaluate,scaling_summary,truth
    from .data import prepare_subset
    params=json.loads(args.params)
    with threadpool_limits(limits=1):
        if args.task=="prepare":return _prepare(config,args.dataset,args.sift_archive)
        if args.task=="subset":return prepare_subset(config,args.count)
        if args.task=="build":return build(config,args.dataset,args.method,params)
        if args.task=="truth":return truth(config,args.dataset)
        if args.task=="evaluate":
            setting="exact" if args.method=="exact" else int(args.setting)
            return evaluate(config,args.dataset,args.method,params,setting)
        if args.task=="scaling-summary":return scaling_summary(config)
    raise ValueError("unknown worker task")


def run(config,args):
    _dependencies()
    deadline=time.time()+args.max_hours*3600
    runtime=ROOT/"cloud/runtime.json"
    if runtime.exists():deadline=min(deadline,read_json(runtime)["termination_epoch"]-180)
    logs=pathlib.Path(config["results_dir"])/"logs";logs.mkdir(parents=True,exist_ok=True)
    sequence=0
    def worker(task,dataset="",method="",params=None,setting="",count=None):
        nonlocal sequence
        remaining=deadline-time.time()
        if remaining<120:
            raise RuntimeError("Runtime allowance exhausted; completed artifacts and build checkpoints are preserved")
        sequence+=1
        command=[sys.executable,"-m","pa2.cli","worker","--config",config["config_path"],"--task",task,
                 "--dataset",dataset,"--method",method,"--params",json.dumps(params or {}),"--setting",str(setting)]
        if count is not None:command += ["--count",str(count)]
        if args.sift_archive and task=="prepare" and dataset=="sift1m":command += ["--sift-archive",args.sift_archive]
        label=f"{sequence:03d}_{task}_{dataset}_{method}_{setting}"
        print(f"Starting {label}; runtime remaining {remaining/3600:.2f} h",flush=True)
        started=time.time()
        with (logs/f"{label}.log").open("a") as output:
            output.write(json.dumps({"command":command,"started_epoch":started})+"\n");output.flush()
            try:
                subprocess.run(command,check=True,stdout=output,stderr=subprocess.STDOUT,timeout=remaining)
            except BaseException:
                print(f"Task failed; inspect {logs/label}.log",file=sys.stderr)
                raise
            finally:
                output.write(json.dumps({"finished_epoch":time.time(),"wall_seconds":time.time()-started})+"\n")
    worker("prepare","sift1m")
    main={k:config["vamana"][k] for k in ("R","L_build","alpha")}
    for count in config["scaling_sizes"]:
        name="sift100k" if count==100000 else f"sift{count}"
        worker("subset",name,count=count)
        worker("build",name,"vamana",main)
    for dataset in ("quora","nq"):worker("prepare",dataset)
    worker("scaling-summary")
    estimate=read_json(pathlib.Path(config["results_dir"])/"scaling.json")
    remaining=deadline-time.time()
    # Leave time for baselines/evaluation/report; do not start infeasible full builds.
    predicted=sum(estimate["estimated_build_seconds"].values())
    if predicted>remaining*.75:
        write_json(pathlib.Path(config["results_dir"])/"feasibility.json",{
            "ready":False,"estimated_main_build_seconds":predicted,"remaining_seconds":remaining,
            "reason":"Estimated full Vamana builds exceed 75% of remaining runtime; optimize/reassess without silently reducing datasets"})
        raise RuntimeError("Scaling predicts insufficient runtime. See results/scaling.json and feasibility.json; no full-dataset sizes or parameters were reduced")
    write_json(pathlib.Path(config["results_dir"])/"feasibility.json",{
        "ready":True,"estimated_main_build_seconds":predicted,"remaining_seconds":remaining})
    for dataset in ["sift100k",*config["full_datasets"]]:
        worker("build",dataset,"exact")
        worker("truth",dataset)
        if dataset in config["full_datasets"]:
            worker("evaluate",dataset,"exact",setting="exact")
            worker("build",dataset,"hnsw",config["hnsw"])
            for setting in config["hnsw"]["efSearch"]:
                worker("evaluate",dataset,"hnsw",config["hnsw"],setting)
    for params in config["study"]:
        worker("build","sift100k","vamana",params)
        for setting in config["vamana"]["L_search"]:
            worker("evaluate","sift100k","vamana",params,setting)
    for dataset in config["full_datasets"]:
        worker("build",dataset,"vamana",main)
        for setting in config["vamana"]["L_search"]:
            worker("evaluate",dataset,"vamana",main,setting)
    from .audit import audit
    summary,_=audit(config,verify_checksums=True)
    print(json.dumps(summary,indent=2))
    if not summary["complete"]:raise RuntimeError("Required experiment audit failed")
    print("Experiments complete. Supply roll numbers/transcripts/share links, then generate the final report and package.")


def core_smoke(args):
    from .core import Vamana
    rng=random.Random(42)
    n,dim=args.points,args.dimensions
    if n<=16 or dim<1 or args.queries<1:raise ValueError("smoke test requires N>16, D>0, Q>0")
    vectors=array.array("f",(rng.randrange(256) for _ in range(n*dim)))
    queries=[array.array("f",(rng.randrange(256) for _ in range(dim))) for _ in range(args.queries)]
    exact=[]
    for query in queries:
        distances=[(sum((vectors[i*dim+j]-query[j])**2 for j in range(dim)),i) for i in range(n)]
        exact.append([i for _,i in sorted(distances)[:10]])
    with Vamana(vectors,16,dim) as index:
        started=time.perf_counter();index.build(30,1.2,42,progress=lambda *args:False)
        build_seconds=time.perf_counter()-started
        points=[]
        for L in sorted({10,min(30,n),min(100,n),n}):
            recalls=[];ones=[];expanded=[];evaluations=[];latencies=[]
            for query,ground in zip(queries,exact):
                start=time.perf_counter();actual=index.search(query,10,L);latencies.append((time.perf_counter()-start)*1000)
                recalls.append(len(set(ground)&set(actual["ids"]))/10);ones.append(actual["ids"][0]==ground[0])
                expanded.append(actual["expanded"]);evaluations.append(actual["distance_evaluations"])
            points.append({"L_search":L,"recall_1":sum(ones)/len(ones),"recall_10":sum(recalls)/len(recalls),
                "mean_latency_ms":sum(latencies)/len(latencies),"expanded_mean":sum(expanded)/len(expanded),
                "distance_evaluations_mean":sum(evaluations)/len(evaluations)})
        result={"kind":"synthetic core validation; NOT assignment experimental results","N":n,"D":dim,"Q":len(queries),
                "seed":42,"build_seconds":build_seconds,"graph":index.info(),"points":points,"machine":machine_info(),"peak_rss_bytes":peak_rss_bytes(),
                "core_source_sha256":sha256(ROOT/"cpp/vamana.cpp")}
    path=pathlib.Path(args.output);write_json(path,result);print(json.dumps(result,indent=2));return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument("--config",default="configs/main.json")
    prepare=sub.add_parser("prepare",parents=[common],help="Download and prepare compulsory datasets")
    prepare.add_argument("--dataset",choices=["all","sift1m","sift100k","quora","nq"],default="all")
    prepare.add_argument("--sift-archive");prepare.add_argument("--force",action="store_true")
    runner=sub.add_parser("run",parents=[common],help="Run the resumable full experiment workflow")
    runner.add_argument("--max-hours",type=float,default=24);runner.add_argument("--sift-archive")
    worker=sub.add_parser("worker",parents=[common],help="Run a single fresh-process experiment task")
    worker.add_argument("--task",required=True,choices=["prepare","subset","build","truth","evaluate","scaling-summary"])
    worker.add_argument("--dataset",default="");worker.add_argument("--method",default="");worker.add_argument("--params",default="{}")
    worker.add_argument("--setting",default="");worker.add_argument("--count",type=int);worker.add_argument("--sift-archive")
    checker=sub.add_parser("audit",parents=[common],help="List missing or invalid required evidence")
    checker.add_argument("--details");checker.add_argument("--include-report",action="store_true");checker.add_argument("--quick",action="store_true",help="Skip large artifact SHA checks")
    report=sub.add_parser("report",parents=[common],help="Generate one PDF report and all plots")
    report.add_argument("--details",default="submission_details.json");report.add_argument("--draft",action="store_true")
    package=sub.add_parser("package",parents=[common],help="Package only after the final completeness audit passes")
    package.add_argument("--details",default="submission_details.json")
    smoke=sub.add_parser("core-smoke",help="Dependency-free synthetic validation, not assignment evidence")
    smoke.add_argument("--points",type=int,default=1000);smoke.add_argument("--dimensions",type=int,default=128)
    smoke.add_argument("--queries",type=int,default=20);smoke.add_argument("--output",default="results/local_core_validation.json")
    args=parser.parse_args(argv);configure_threads()
    try:
        if args.command=="core-smoke":core_smoke(args);return 0
        config=load_config(args.config)
        if args.command=="prepare":
            _dependencies()
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):_prepare(config,args.dataset,args.sift_archive,args.force)
        elif args.command=="worker":_worker(config,args)
        elif args.command=="run":run(config,args)
        elif args.command=="audit":
            from .audit import audit
            summary,_=audit(config,args.details,args.include_report,not args.quick)
            print(json.dumps(summary,indent=2));return 0 if summary["complete"] else 2
        elif args.command=="report":
            from .report import generate
            print(generate(config,args.details,args.draft))
        elif args.command=="package":
            from .report import package
            print(package(config,args.details))
        return 0
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        print(f"Error: {error}",file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
