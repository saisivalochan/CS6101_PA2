"""Fresh-process builds, faithful recall evaluation, and consistent CPU timing."""
import importlib.metadata
import pathlib
import statistics
import time
import uuid

from . import data
from .core import Vamana, ROOT
from .util import (configure_threads, fingerprint, index_id, machine_info, peak_rss_bytes,
                   read_json, sha256, write_json)


def locations(config,dataset,method,params):
    name=index_id(method,params)
    directory=pathlib.Path(config["index_dir"])/dataset;directory.mkdir(parents=True,exist_ok=True)
    results=pathlib.Path(config["results_dir"])
    builds=results/"builds";builds.mkdir(parents=True,exist_ok=True)
    return name,directory/f"{name}.{'vamana' if method=='vamana' else 'faiss'}",builds/f"{dataset}__{name}.json"


def _faiss():
    configure_threads()
    import faiss
    faiss.omp_set_num_threads(1)
    return faiss


def build_spec(info,method,params,seed):
    critical=({k:params[k] for k in ("R","L_build","alpha")} if method=="vamana" else
              {k:params[k] for k in ("M","efConstruction")} if method=="hnsw" else {})
    implementation=(sha256(ROOT/"cpp/vamana.cpp") if method=="vamana" else importlib.metadata.version("faiss-cpu"))
    return {"dataset_fingerprint":info["fingerprint"],"method":method,"parameters":critical,
            "seed":seed,"implementation":implementation,"metric":"L2"}


def build(config,dataset,method,params):
    configure_threads()
    import numpy as np
    faiss=_faiss()
    base,_,info=data.load_vectors(config,dataset)
    name,path,meta_path=locations(config,dataset,method,params)
    spec=build_spec(info,method,params,config["seed"])
    signature=fingerprint(spec)
    if meta_path.exists():
        metadata=read_json(meta_path)
        if metadata["signature"]!=signature:
            raise ValueError(f"stale index {name}; move incompatible index/metadata aside before rebuilding")
        if not path.exists() or sha256(path)!=metadata["index_sha256"]:
            raise ValueError(f"missing or corrupted persisted index: {path}")
        print(f"Reusing {dataset}/{name}",flush=True)
        return metadata
    started=time.perf_counter()
    extra={}
    if method=="vamana":
        checkpoint=path.with_suffix(".checkpoint")
        checkpoint_meta=path.with_suffix(".checkpoint.json")
        if checkpoint_meta.exists() and read_json(checkpoint_meta)["signature"]!=signature:
            raise ValueError("checkpoint provenance mismatch")
        if checkpoint.exists() and not checkpoint_meta.exists():
            raise ValueError("checkpoint is missing its provenance")
        write_json(checkpoint_meta,{"signature":signature,"spec":spec})
        attempts_dir=pathlib.Path(config["results_dir"])/"build_attempts"
        attempt_path=attempts_dir/f"{dataset}__{name}__{uuid.uuid4().hex}.json"
        attempt={"signature":signature,"dataset":dataset,"index_id":name,"started_epoch":time.time(),
                 "construction_seconds":0.0,"peak_rss_bytes":peak_rss_bytes(),"finished":False}
        def heartbeat(finished=False):
            attempt.update({"construction_seconds":time.perf_counter()-started,"peak_rss_bytes":peak_rss_bytes(),"finished":finished})
            write_json(attempt_path,attempt)
        heartbeat()
        try:
            with Vamana(base,params["R"]) as index:
                if checkpoint.exists():
                    index.load(checkpoint)
                progress_path=pathlib.Path(config["results_dir"])/"progress"/f"{dataset}__{name}.json"
                def progress(pass_number,processed,total,seconds):
                    value={"dataset":dataset,"index":name,"pass":pass_number,"processed":processed,
                           "total":total,"build_seconds":seconds,"peak_rss_bytes":peak_rss_bytes()}
                    write_json(progress_path,value);heartbeat()
                    print(value,flush=True)
                    return False
                index.build(params["L_build"],params["alpha"],config["seed"],checkpoint,
                            config["checkpoint_seconds"],progress)
                extra=index.info();index.validate()
                heartbeat(True)
                # Serialization time is recorded separately from construction.
                index.save(path)
        except BaseException:
            if not attempt["finished"]:heartbeat(True)
            raise
        native_seconds=extra.pop("build_seconds")
        attempts=[read_json(p) for p in attempts_dir.glob(f"{dataset}__{name}__*.json")]
        attempts=[a for a in attempts if a["signature"]==signature]
        build_seconds=sum(a["construction_seconds"] for a in attempts)
        extra.update({"successful_pass_seconds":native_seconds,"attempt_count":len(attempts),
                      "attempt_timing_lower_bound":any(not a["finished"] for a in attempts),
                      "peak_rss_all_attempts_bytes":max(a["peak_rss_bytes"] for a in attempts)})
        extra.update({"graph_representation":"uint32 fixed-width adjacency plus uint32 degrees",
                      "entry_procedure":"database point nearest corpus mean (float64 mean and selection)",
                      "validation":"all IDs valid; no self/duplicate edges; every degree <= R",
                      "graph_bytes":path.stat().st_size})
        # Keep the final graph; the completed checkpoint is redundant.
        checkpoint.unlink(missing_ok=True)
        checkpoint_meta.unlink(missing_ok=True)
    elif method in ("exact","hnsw"):
        if method=="exact":
            index=faiss.IndexFlatL2(128)
        else:
            index=faiss.IndexHNSWFlat(128,params["M"],faiss.METRIC_L2)
            index.hnsw.efConstruction=params["efConstruction"]
            # FAISS initializes its HNSW RNG deterministically (default seed 12345).
            extra["hnsw_rng_seed"]=12345
        for offset in range(0,len(base),65536):
            index.add(np.ascontiguousarray(base[offset:offset+65536]))
        build_seconds=time.perf_counter()-started
        temporary=path.with_name(path.name+".partial")
        faiss.write_index(index,str(temporary));temporary.replace(path)
    else:
        raise ValueError(f"unknown method {method}")
    metadata={"dataset":dataset,"index_id":name,"method":method,"signature":signature,"spec":spec,
              "build_seconds":build_seconds,"build_process_peak_rss_bytes":max(peak_rss_bytes(),extra.get("peak_rss_all_attempts_bytes",0)),
              "serialization_and_current_process_seconds":time.perf_counter()-started,
              "index_bytes":path.stat().st_size+(data.paths(config,dataset)/"corpus.npy").stat().st_size if method=="vamana" else path.stat().st_size,
              "serialized_bytes":path.stat().st_size,"index_sha256":sha256(path),"machine":machine_info(),**extra}
    write_json(meta_path,metadata)
    return metadata


def truth(config,dataset):
    import numpy as np
    faiss=_faiss()
    base,queries,info=data.load_vectors(config,dataset)
    directory=pathlib.Path(config["index_dir"])/dataset
    path=directory/"truth.npz";meta=directory/"truth.json"
    if path.exists() and meta.exists():
        provenance=read_json(meta)
        if provenance["dataset_fingerprint"]!=info["fingerprint"] or provenance["sha256"]!=sha256(path):
            raise ValueError("stale/corrupt exact-neighbor cache")
        write_json(pathlib.Path(config["results_dir"])/"truth"/f"{dataset}.json",provenance)
        return provenance
    _,index_path,build_meta=locations(config,dataset,"exact",{})
    if not index_path.exists() or not build_meta.exists():
        raise ValueError("build the exact FAISS index before computing truth")
    exact_build=read_json(build_meta)
    if exact_build["spec"]["dataset_fingerprint"]!=info["fingerprint"] or sha256(index_path)!=exact_build["index_sha256"]:
        raise ValueError("stale exact index")
    exact=faiss.read_index(str(index_path))
    ids=np.empty((len(queries),10),dtype=np.int64);distances=np.empty_like(ids,dtype=np.float32)
    for offset in range(0,len(queries),128):
        d,i=exact.search(queries[offset:offset+128],10)
        distances[offset:offset+len(i)]=d;ids[offset:offset+len(i)]=i
    provenance={"dataset_fingerprint":info["fingerprint"],"metric":"L2","k":10,"query_count":len(queries),
                "faiss_version":faiss.__version__,"reference":"projected-space exact FAISS"}
    reference_ids=ids
    if dataset=="sift1m":
        supplied=np.load(data.paths(config,dataset)/"supplied_truth.npy",allow_pickle=False)
        direct=np.sum((base[supplied]-queries[:,None,:])**2,axis=2)
        discrepancy=np.abs(np.sort(direct,axis=1)-distances)
        if not np.allclose(np.sort(direct,axis=1),distances,rtol=1e-5,atol=1e-4):
            raise ValueError("FAISS SIFT1M distances disagree with supplied ground truth")
        provenance.update({"reference":"supplied SIFT1M IDs","faiss_exact_id_agreement":float(np.mean(ids==supplied)),
            "supplied_distances_verified":True,"maximum_supplied_distance_difference":float(discrepancy.max()),
            "tie_note":"ID differences at identical distances are retained and evaluated against supplied IDs"})
        reference_ids=supplied
    temporary=path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez(stream,ids=reference_ids,faiss_ids=ids,distances=distances)
    temporary.replace(path)
    provenance["sha256"]=sha256(path);write_json(meta,provenance)
    # Include compact verification evidence in the result bundle/report inputs.
    write_json(pathlib.Path(config["results_dir"])/"truth"/f"{dataset}.json",provenance)
    return provenance


def recall(ids,ground_truth):
    """Literal nearest-neighbor fidelity, with sets removing duplicate outputs."""
    if len(ids)!=len(ground_truth) or not len(ids):
        raise ValueError("recall needs equally sized nonempty query sets")
    one=sum(int(a[0])==int(g[0]) for a,g in zip(ids,ground_truth))/len(ids)
    ten=sum(len(set(map(int,a[:10])) & set(map(int,g[:10])))/10 for a,g in zip(ids,ground_truth))/len(ids)
    return {"recall_1":one,"recall_10":ten}


def _timing_rows(config,dataset,query_count):
    import numpy as np
    path=pathlib.Path(config["results_dir"])/"timing"/f"{dataset}_rows.json"
    rows=sorted(np.random.default_rng(config["seed"]).choice(query_count,min(query_count,config["timing_queries"]),replace=False).tolist())
    value={"seed":config["seed"],"full_query_count":query_count,"row_indices":rows}
    if path.exists() and read_json(path)!=value:
        raise ValueError("timing-query policy changed; archive old results before rerunning")
    write_json(path,value)
    return rows


def _time_queries(call,queries,rows,warmup,repeats):
    for row in rows[:warmup]:
        call(queries[row])
    runs=[]
    for _ in range(repeats):
        latencies=[]
        for row in rows:
            query=queries[row]
            start=time.perf_counter_ns();call(query);end=time.perf_counter_ns()
            latencies.append((end-start)/1e6)
        mean=statistics.fmean(latencies)
        runs.append({"mean_latency_ms":mean,"qps":1000/mean,"latencies_ms":latencies})
    selected=sorted(range(len(runs)),key=lambda i:runs[i]["mean_latency_ms"])[len(runs)//2]
    return runs,selected


def evaluate(config,dataset,method,params,search_parameter):
    import numpy as np
    faiss=_faiss()
    base,queries,info=data.load_vectors(config,dataset)
    name,path,build_path=locations(config,dataset,method,params)
    metadata=read_json(build_path)
    if metadata["signature"]!=fingerprint(build_spec(info,method,params,config["seed"])):
        raise ValueError("index provenance mismatch")
    ground_path=pathlib.Path(config["index_dir"])/dataset/"truth.npz"
    ground_meta=read_json(ground_path.with_suffix(".json"))
    if ground_meta["dataset_fingerprint"]!=info["fingerprint"] or ground_meta["sha256"]!=sha256(ground_path):
        raise ValueError("ground truth provenance mismatch")
    with np.load(ground_path,allow_pickle=False) as saved:
        ground=saved["ids"];faiss_exact=saved["faiss_ids"]
    point_id=f"{dataset}__{name}__{search_parameter}"
    result_path=pathlib.Path(config["results_dir"])/"points"/f"{point_id}.json"
    policy={"threads":1,"query_mode":"individual","k":10,"index_memory":"resident; loading excluded","timing_queries":config["timing_queries"],
            "repeats":config["timing_repeats"],"warmup_queries":config["warmup_queries"],"seed":config["seed"]}
    evaluation_implementation={name:sha256(ROOT/"pa2"/name) for name in ("core.py","experiments.py")}
    expected=fingerprint({"index_signature":metadata["signature"],"index_sha256":metadata["index_sha256"],
                          "search_parameter":search_parameter,"policy":policy,"truth":ground_meta["sha256"],
                          "evaluation_implementation":evaluation_implementation})
    if result_path.exists():
        cached=read_json(result_path)
        if cached["signature"]!=expected:
            raise ValueError("stale operating-point result")
        timing_file=pathlib.Path(config["results_dir"])/"timing"/f"{point_id}.json"
        if timing_file.exists():
            print(f"Reusing operating point {point_id}",flush=True)
            return cached
    if sha256(path)!=metadata["index_sha256"]:
        raise ValueError("index checksum mismatch")
    output=np.empty((len(queries),10),dtype=np.int64)
    expanded=0;evaluations=0
    index=None
    try:
        if method=="vamana":
            # Materialize the mapped corpus before searching. FAISS read_index
            # loads its stored vectors; Vamana must likewise exclude first-touch
            # disk reads from the in-memory latency comparison.
            base=np.array(base,dtype=np.float32,order="C",copy=True)
            index=Vamana(base,params["R"]).load(path)
            def call(query):
                return index.search_raw(query,10,int(search_parameter))
            for row,query in enumerate(queries):
                ids,_,stats,_=call(query)
                output[row]=[int(i) if i!=2**32-1 else -1 for i in ids]
                expanded+=int(stats[0]);evaluations+=int(stats[1])
        else:
            index=faiss.read_index(str(path))
            if method=="hnsw":
                index.hnsw.efSearch=int(search_parameter)
            def call(query):
                return index.search(query.reshape(1,-1),10)
            if method=="exact":
                output[:]=faiss_exact
            else:
                for row,query in enumerate(queries):
                    output[row]=call(query)[1][0]
        quality=recall(output,ground)
        timing_rows=_timing_rows(config,dataset,len(queries))
        runs,selected=_time_queries(call,queries,timing_rows,config["warmup_queries"],config["timing_repeats"])
        timing_path=pathlib.Path(config["results_dir"])/"timing"/f"{point_id}.json"
        write_json(timing_path,{"policy":policy,"runs":runs,"selected_run":selected,"selection":"median run by mean latency"})
        result={"dataset":dataset,"method":method,"index_id":name,"search_parameter":search_parameter,
                "signature":expected,"index_signature":metadata["signature"],"dataset_fingerprint":info["fingerprint"],
                **quality,"mean_latency_ms":runs[selected]["mean_latency_ms"],"qps":runs[selected]["qps"],
                "build_seconds":metadata["build_seconds"],"index_bytes":metadata["index_bytes"],
                "distance_evaluations_per_query":evaluations/len(queries) if method=="vamana" else None,
                "vertices_expanded_per_query":expanded/len(queries) if method=="vamana" else None,
                "evaluation_queries":len(queries),"timing_query_count":len(timing_rows),"policy":policy,
                "parameters":metadata["spec"]["parameters"],"machine":machine_info()}
        if method=="vamana":
            result.update({"R":params["R"],"L_build":params["L_build"],"alpha":params["alpha"],"L_search":search_parameter})
            if dataset=="sift100k" and all(params[k]==config["vamana"][k] for k in ("R","L_build","alpha")) and search_parameter==100:
                trace=index.search(queries[0],10,100,True)
                trace.update({"query_row":0,"exact_top10_ids":ground[0].tolist(),
                              "recall_10":len(set(trace["ids"])&set(map(int,ground[0])))/10,
                              "id_space":"row IDs in the SIFT100K subset","dataset_fingerprint":info["fingerprint"]})
                write_json(pathlib.Path(config["results_dir"])/"sift100k_trace.json",trace)
        elif method=="hnsw":
            result["efSearch"]=search_parameter
        write_json(result_path,result)
        print({k:result[k] for k in ("dataset","method","search_parameter","recall_1","recall_10","mean_latency_ms")},flush=True)
        return result
    finally:
        if isinstance(index,Vamana):
            index.close()


def scaling_summary(config):
    points=[]
    params={k:config["vamana"][k] for k in ("R","L_build","alpha")}
    for count in config["scaling_sizes"]:
        dataset="sift100k" if count==100000 else f"sift{count}"
        _,_,path=locations(config,dataset,"vamana",params)
        value=read_json(path)
        points.append({"points":count,"build_seconds":value["build_seconds"],
                       "peak_build_rss_bytes":value["build_process_peak_rss_bytes"],"signature":value["signature"]})
    # Conservative empirical extrapolation, not a promise for a different dataset.
    import math
    exponent=max(1.0,math.log(points[-1]["build_seconds"]/points[0]["build_seconds"])/math.log(points[-1]["points"]/points[0]["points"]))
    estimates={}
    for dataset in config["full_datasets"]:
        info=data.manifest(config,dataset)
        estimates[dataset]=points[-1]["build_seconds"]*(info["corpus_size"]/points[-1]["points"])**exponent
    value={"measurements":points,"empirical_exponent":exponent,"estimated_build_seconds":estimates,
           "limitation":"SIFT scaling extrapolation; Quora/NQ geometry and build cost may differ materially"}
    write_json(pathlib.Path(config["results_dir"])/"scaling.json",value)
    print(value,flush=True)
    return value
