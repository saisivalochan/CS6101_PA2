"""Stream public embeddings, prepare search vectors, and persist provenance."""
import json
import pathlib
import shutil
import tarfile
import urllib.request

from .util import fingerprint, read_json, sha256, write_json

HF_SOURCE = "CohereLabs/beir-embed-english-v3"
SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"


def paths(config, dataset):
    return pathlib.Path(config["data_dir"]) / dataset


def manifest(config, dataset, verify=False):
    directory=paths(config,dataset)
    value=read_json(directory/"manifest.json")
    expected=value["fingerprint"]
    if fingerprint({k:v for k,v in value.items() if k!="fingerprint"})!=expected:
        raise ValueError(f"manifest fingerprint mismatch: {dataset}")
    for name, description in value["artifacts"].items():
        path=directory/name
        if not path.is_file() or path.stat().st_size!=description["bytes"]:
            raise ValueError(f"missing or changed prepared artifact: {path}")
        if verify and sha256(path)!=description["sha256"]:
            raise ValueError(f"prepared checksum mismatch: {path}")
    return value


def _finish(directory, value, artifacts):
    value["artifacts"]={name:{"bytes":(directory/name).stat().st_size,"sha256":sha256(directory/name)} for name in artifacts}
    value["fingerprint"]=fingerprint(value)
    write_json(directory/"manifest.json",value)
    return value


def _save_array(path, values):
    import numpy as np
    temporary=path.with_name(path.name+".partial")
    with temporary.open("wb") as stream:
        np.save(stream,values,allow_pickle=False)
    temporary.replace(path)


def _vectors(path, kind):
    import numpy as np
    raw=np.memmap(path,dtype="<i4",mode="r")
    if len(raw)==0:
        raise ValueError("empty vector file")
    dimension=int(raw[0])
    if dimension<1 or len(raw)%(dimension+1):
        raise ValueError("invalid fvecs/ivecs record layout")
    rows=raw.reshape(-1,dimension+1)
    if not np.all(rows[:,0]==dimension):
        raise ValueError("inconsistent fvecs/ivecs dimensions")
    return rows[:,1:].view("<f4") if kind=="f" else rows[:,1:]


def prepare_sift(config, archive=None, force=False):
    import numpy as np
    directory=paths(config,"sift1m");directory.mkdir(parents=True,exist_ok=True)
    if (directory/"manifest.json").exists() and not force:
        return manifest(config,"sift1m",verify=True)
    raw=pathlib.Path(config["data_dir"])/"raw/sift";raw.mkdir(parents=True,exist_ok=True)
    archive=pathlib.Path(archive) if archive else raw/"sift.tar.gz"
    if not archive.exists():
        temporary=archive.with_name(archive.name+".partial")
        with urllib.request.urlopen(SIFT_URL,timeout=120) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source,target,8*1024*1024)
        temporary.replace(archive)
    filenames=["sift_base.fvecs","sift_query.fvecs","sift_groundtruth.ivecs"]
    with tarfile.open(archive,"r:gz") as package:
        for name in filenames:
            destination=raw/name
            if not destination.exists():
                members=[m for m in package.getmembers() if pathlib.PurePosixPath(m.name).name==name and m.isfile()]
                if len(members)!=1:
                    raise ValueError(f"missing/ambiguous SIFT member: {name}")
                with package.extractfile(members[0]) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source,target)
    base=_vectors(raw/filenames[0],"f");queries=_vectors(raw/filenames[1],"f");truth=_vectors(raw/filenames[2],"i")
    if base.shape!=(1000000,128) or queries.shape[1]!=128 or truth.shape[0]!=len(queries) or truth.shape[1]<10:
        raise ValueError("SIFT1M archive does not match the standard benchmark")
    if not np.isfinite(base).all() or not np.isfinite(queries).all():
        raise ValueError("SIFT contains nonfinite vectors")
    if truth.min()<0 or truth.max()>=len(base):
        raise ValueError("invalid supplied SIFT neighbor IDs")
    _save_array(directory/"corpus.npy",np.ascontiguousarray(base,dtype=np.float32))
    _save_array(directory/"queries.npy",np.ascontiguousarray(queries,dtype=np.float32))
    _save_array(directory/"supplied_truth.npy",np.ascontiguousarray(truth[:,:10],dtype=np.int64))
    _save_array(directory/"corpus_ids.npy",np.arange(len(base),dtype=np.int64))
    return _finish(directory,{"dataset":"sift1m","corpus_size":len(base),"query_count":len(queries),"original_dimension":128,
        "search_dimension":128,"metric":"L2","normalized":False,"raw_corpus_bytes":int(base.size*4),
        "source":SIFT_URL,"source_archive_sha256":sha256(archive),"ground_truth":"supplied SIFT1M"},
        ["corpus.npy","queries.npy","supplied_truth.npy","corpus_ids.npy"])


def prepare_subset(config, count, name=None, force=False):
    import numpy as np
    name=name or ("sift100k" if count==100000 else f"sift{count}")
    parent=manifest(config,"sift1m")
    directory=paths(config,name);directory.mkdir(parents=True,exist_ok=True)
    if (directory/"manifest.json").exists() and not force:
        existing=manifest(config,name,verify=True)
        if existing.get("parent_fingerprint")!=parent["fingerprint"] or existing.get("seed")!=config["seed"]:
            raise ValueError(f"stale subset {name}; prepare with --force")
        return existing
    source=np.load(paths(config,"sift1m")/"corpus.npy",mmap_mode="r")
    if not 10<=count<=len(source):
        raise ValueError("invalid SIFT subset size")
    selected=np.sort(np.random.default_rng(config["seed"]).choice(len(source),count,replace=False))
    _save_array(directory/"corpus.npy",np.ascontiguousarray(source[selected]))
    shutil.copyfile(paths(config,"sift1m")/"queries.npy",directory/"queries.npy")
    _save_array(directory/"corpus_ids.npy",selected.astype(np.int64))
    return _finish(directory,{"dataset":name,"corpus_size":count,"query_count":parent["query_count"],"original_dimension":128,
        "search_dimension":128,"metric":"L2","normalized":False,"raw_corpus_bytes":count*128*4,
        "seed":config["seed"],"parent_fingerprint":parent["fingerprint"],"ground_truth":"new exact FAISS on this subset"},
        ["corpus.npy","queries.npy","corpus_ids.npy"])


def _stream_raw(config, dataset, collection, revision):
    import numpy as np
    from datasets import load_dataset
    raw=pathlib.Path(config["data_dir"])/"raw"/dataset;raw.mkdir(parents=True,exist_ok=True)
    binary=raw/f"{collection}.f32";ids=raw/f"{collection}_ids.jsonl";meta=raw/f"{collection}.json"
    if meta.exists():
        info=read_json(meta)
        if info["revision"]!=revision or not binary.exists() or binary.stat().st_size!=info["count"]*1024*4 or not ids.exists():
            raise ValueError("raw cache provenance mismatch; remove this dataset's raw cache before retrying")
        return binary,ids,info["count"]
    split="train" if collection=="corpus" else "test"
    source=load_dataset(HF_SOURCE,f"{dataset}-{collection}",split=split,revision=revision,
                        streaming=True,cache_dir=str(pathlib.Path(config["data_dir"])/"hf-cache"))
    source=source.select_columns(["_id","emb"])
    count=0
    temporary=binary.with_suffix(".partial");idtemp=ids.with_suffix(".partial")
    with temporary.open("wb") as output,idtemp.open("w") as identifiers:
        for batch in source.iter(batch_size=2048):
            vectors=np.asarray(batch["emb"],dtype=np.float32)
            if vectors.ndim!=2 or vectors.shape[1]!=1024 or len(vectors)!=len(batch["_id"]) or not np.isfinite(vectors).all():
                raise ValueError("invalid Cohere embedding batch")
            vectors.tofile(output)
            for identifier in batch["_id"]:
                identifiers.write(json.dumps(str(identifier))+"\n")
            count+=len(vectors)
            if count%100352==0:
                print(f"{dataset}/{collection}: {count:,} rows",flush=True)
    if count==0:
        raise ValueError("empty embedding collection")
    temporary.replace(binary);idtemp.replace(ids)
    write_json(meta,{"source":HF_SOURCE,"revision":revision,"split":split,"count":count,"dimension":1024})
    return binary,ids,count


def prepare_ir(config,dataset,force=False):
    import numpy as np
    import sklearn
    from sklearn.decomposition import PCA
    from huggingface_hub import HfApi
    if dataset not in ("quora","nq"):
        raise ValueError("only Quora and NQ are compulsory IR datasets")
    directory=paths(config,dataset);directory.mkdir(parents=True,exist_ok=True)
    if (directory/"manifest.json").exists() and not force:
        value=manifest(config,dataset,verify=True)
        if value["pca"]["seed"]!=config["seed"] or value["pca"]["sample_size"]!=min(config["pca_sample"],value["corpus_size"]):
            raise ValueError("PCA config changed; prepare with --force")
        return value
    # Pin the first resolved revision even if upstream changes during a retry.
    revision_file=directory/"source.json"
    revision=read_json(revision_file)["revision"] if revision_file.exists() else HfApi().dataset_info(HF_SOURCE).sha
    write_json(revision_file,{"source":HF_SOURCE,"revision":revision})
    corpus_raw,corpus_ids,n=_stream_raw(config,dataset,"corpus",revision)
    query_raw,query_ids,nq=_stream_raw(config,dataset,"queries",revision)
    corpus=np.memmap(corpus_raw,dtype=np.float32,mode="r",shape=(n,1024))
    queries=np.memmap(query_raw,dtype=np.float32,mode="r",shape=(nq,1024))
    selected=np.random.default_rng(config["seed"]).choice(n,min(config["pca_sample"],n),replace=False)
    if len(selected)<128:
        raise ValueError("PCA fitting sample needs at least 128 corpus rows")
    estimator=PCA(n_components=128,svd_solver="randomized",random_state=config["seed"])
    estimator.fit(np.ascontiguousarray(corpus[selected]))
    with (directory/"pca.npz").open("wb") as stream:
        np.savez(stream,mean=estimator.mean_,components=estimator.components_,
                 explained_variance_ratio=estimator.explained_variance_ratio_,sample_rows=selected)
    for name,vectors in (("corpus",corpus),("queries",queries)):
        temporary=directory/f"{name}.npy.partial"
        target=np.lib.format.open_memmap(temporary,mode="w+",dtype=np.float32,shape=(len(vectors),128))
        for start in range(0,len(vectors),8192):
            projected=estimator.transform(vectors[start:start+8192]).astype(np.float32)
            norms=np.linalg.norm(projected,axis=1,keepdims=True)
            if not np.isfinite(projected).all() or not np.isfinite(norms).all() or (norms==0).any():
                raise ValueError("zero/nonfinite projected vector; cannot normalize")
            target[start:start+len(projected)]=projected/norms
        target.flush();del target
        temporary.replace(directory/f"{name}.npy")
    shutil.copyfile(corpus_ids,directory/"corpus_ids.jsonl")
    shutil.copyfile(query_ids,directory/"query_ids.jsonl")
    return _finish(directory,{"dataset":dataset,"corpus_size":n,"query_count":nq,"original_dimension":1024,
        "search_dimension":128,"metric":"L2","normalized":True,"raw_corpus_bytes":n*128*4,
        "source":HF_SOURCE,"revision":revision,"query_split":"test","ground_truth":"exact FAISS after PCA and normalization",
        "pca":{"implementation":"sklearn.decomposition.PCA(randomized)","sklearn_version":sklearn.__version__,
            "sample_size":len(selected),"seed":config["seed"],"fitting_source":"corpus only",
            "explained_variance":float(estimator.explained_variance_ratio_.sum())}},
        ["corpus.npy","queries.npy","corpus_ids.jsonl","query_ids.jsonl","pca.npz"])


def load_vectors(config,dataset):
    import numpy as np
    info=manifest(config,dataset)
    directory=paths(config,dataset)
    base=np.load(directory/"corpus.npy",mmap_mode="r",allow_pickle=False)
    queries=np.load(directory/"queries.npy",mmap_mode="r",allow_pickle=False)
    if base.shape!=(info["corpus_size"],128) or queries.shape!=(info["query_count"],128):
        raise ValueError("prepared vector shape mismatch")
    if base.dtype!=np.float32 or queries.dtype!=np.float32 or not base.flags.c_contiguous or not queries.flags.c_contiguous:
        raise ValueError("invalid search-vector representation")
    return base,queries,info
