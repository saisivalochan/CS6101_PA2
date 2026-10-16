"""Generate all figures, a single evidence-based PDF, and the submission archive."""
import html
import pathlib
import re
import statistics
import zipfile

from .audit import audit,submission_details
from .util import ROOT,fingerprint,read_json,save_csv,sha256,write_json


def transcript_attachments(text, directory):
    """Resolve only checksum-named image attachments emitted by the exporter."""
    pattern=r"\[Binary attachment: (attachments/([0-9a-f]{64})\.(?:png|jpg|webp)); media=image/(?:png|jpeg|webp); sha256=([0-9a-f]{64})\]"
    paths=[]
    for relative,filename_hash,recorded_hash in re.findall(pattern,text):
        path=pathlib.Path(directory)/relative
        if filename_hash!=recorded_hash or sha256(path)!=recorded_hash:
            raise ValueError(f"Transcript attachment checksum mismatch: {relative}")
        paths.append(path)
    return paths


def _plots(config,evidence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=pathlib.Path(config["results_dir"]);out=root/"figures";out.mkdir(parents=True,exist_ok=True)
    points=evidence["points"];figures=[]
    for dataset in config["full_datasets"]:
        for combined in (False,True):
            selected=[p for p in points if p["dataset"]==dataset and (combined or p["method"]=="hnsw")]
            if not selected:continue
            fig,ax=plt.subplots(figsize=(6.5,4.2))
            for method in ("hnsw","vamana","exact") if combined else ("hnsw",):
                series=sorted([p for p in selected if p["method"]==method],key=lambda p:p["mean_latency_ms"])
                if series:
                    ax.plot([p["mean_latency_ms"] for p in series],[p["recall_10"] for p in series],
                            marker="*" if method=="exact" else "o",linestyle="None" if method=="exact" else "-",label=method.upper())
            ax.set(xlabel="Mean latency per query (ms)",ylabel="10-recall@10",title=f"{dataset}: {'method comparison' if combined else 'FAISS HNSW'}",ylim=(0,1.02))
            if combined:
                ax.set_xscale("log")
                ax.set_xlabel("Mean latency per query (ms; logarithmic axis)")
            ax.grid(alpha=.25);ax.legend();fig.tight_layout()
            path=out/f"{dataset}_{'comparison' if combined else 'hnsw'}.png";fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
    development=[p for p in points if p["dataset"]=="sift100k" and p["method"]=="vamana"]
    for x,label,suffix in (("L_search","L_search","search_list"),("mean_latency_ms","Mean latency per query (ms)","latency")):
        if not development:continue
        fig,ax=plt.subplots(figsize=(6.5,4.2))
        for name in sorted({p["index_id"] for p in development}):
            series=sorted([p for p in development if p["index_id"]==name],key=lambda p:p[x])
            ax.plot([p[x] for p in series],[p["recall_10"] for p in series],"o-",label=name)
        ax.set(xlabel=label,ylabel="10-recall@10",title="SIFT100K parameter study",ylim=(0,1.02));ax.grid(alpha=.25);ax.legend(fontsize=7);fig.tight_layout()
        path=out/f"sift100k_{suffix}.png";fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
    return figures


def comparable(points,dataset):
    selected=[p for p in points if p["dataset"]==dataset]
    exact=[p for p in selected if p["method"]=="exact"]
    hnsw=[p for p in selected if p["method"]=="hnsw"]
    vamana=[p for p in selected if p["method"]=="vamana"]
    if not hnsw or not vamana:return exact,None
    h,v=min(((h,v) for h in hnsw for v in vamana),key=lambda pair:(abs(pair[0]["recall_10"]-pair[1]["recall_10"]),-min(pair[0]["recall_10"],pair[1]["recall_10"])))
    return exact+[h,v],abs(h["recall_10"]-v["recall_10"])


def generate(config,details_path,draft=False):
    summary,evidence=audit(config,details_path,verify_checksums=True)
    if not draft and not summary["complete"]:
        raise ValueError("Cannot generate a final report: "+"; ".join(summary["missing_or_invalid"]))
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,Preformatted
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import ImageReader
    import matplotlib
    font_dir=pathlib.Path(matplotlib.get_data_path())/"fonts/ttf"
    pdfmetrics.registerFont(TTFont("PA2Sans",str(font_dir/"DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("PA2SansBold",str(font_dir/"DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("PA2Sans",normal="PA2Sans",bold="PA2SansBold",italic="PA2Sans",boldItalic="PA2SansBold")
    root=pathlib.Path(config["results_dir"]);root.mkdir(parents=True,exist_ok=True)
    figures=_plots(config,evidence);styles=getSampleStyleSheet();story=[]
    for style in styles.byName.values():
        style.fontName="PA2SansBold" if style.name.startswith("Heading") or style.name=="Title" else "PA2Sans"
    def paragraph(text,style="BodyText"):
        story.append(Paragraph(html.escape(str(text)).replace("\n","<br/>"),styles[style]));story.append(Spacer(1,5))
    def heading(text):paragraph(text,"Heading2")
    def table(headers,rows):
        if not rows:
            paragraph("No measured results available.");return
        content=[[Paragraph(html.escape(str(h)),styles["BodyText"]) for h in headers]]
        content += [[Paragraph(html.escape(str(cell)),styles["BodyText"]) for cell in row] for row in rows]
        item=Table(content,repeatRows=1,hAlign="LEFT",colWidths=[6.8*inch/len(headers)]*len(headers))
        item.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e5e7eb")),("GRID",(0,0),(-1,-1),.3,colors.grey),
                                  ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4)]))
        story.extend([item,Spacer(1,9)])
    def image(name):
        path=root/"figures"/name
        if path in figures:story.extend([Image(str(path),width=6.5*inch,height=4.2*inch),Spacer(1,8)])
    def format_number(value):
        return "—" if value is None else f"{value:.5g}" if isinstance(value,float) else str(value)
    def metric_table(points,counters=False):
        headers=["Method / setting","1-recall@1","10-recall@10","Mean ms","QPS","Build s","Index MiB"]
        rows=[]
        for p in points:
            rows.append([f"{p['index_id']} / {p['search_parameter']}",f"{p['recall_1']:.4f}",f"{p['recall_10']:.4f}",
                         f"{p['mean_latency_ms']:.4f}",f"{p['qps']:.2f}",f"{p['build_seconds']:.2f}",f"{p['index_bytes']/2**20:.2f}"])
        table(headers,rows)
        if counters:
            table(["Vamana setting","Distances/query","Expansions/query"],[[f"{p['index_id']} / {p['search_parameter']}",format_number(p["distance_evaluations_per_query"]),format_number(p["vertices_expanded_per_query"])] for p in points if p["method"]=="vamana"])
    paragraph("CS6101 PA2 — Approximate Nearest Neighbor Search","Title")
    details,_=submission_details(details_path)
    paragraph("Roll numbers: "+", ".join(details.get("roll_numbers",[])))
    if draft:
        heading("INCOMPLETE DRAFT — not a submission")
        paragraph("Missing or invalid evidence:\n"+"\n".join(summary["missing_or_invalid"]))
    heading("Dataset preparation")
    table(["Dataset","Corpus N","Queries","Original D","Search D","Raw float32 MiB"],
          [[d,m["corpus_size"],m["query_count"],m["original_dimension"],128,f"{m['raw_corpus_bytes']/2**20:.2f}"] for d,m in evidence["datasets"].items() if d in config["full_datasets"]])
    paragraph("SIFT uses the supplied 128-dimensional L2 vectors without normalization. Quora and NQ each use a separate corpus-only randomized PCA, applied unchanged to corpus and test queries, cast to float32, then L2-normalized. All methods use L2 on the resulting vectors. BEIR relevance judgments are not the ANN ground truth.")
    for d,m in evidence["datasets"].items():
        if "pca" in m:paragraph(f"{d}: {m['pca']}. Source revision: {m['revision']}.")
    heading("Experimental protocol")
    paragraph("Recall is mean set overlap with the exact top-k, divided by k, for k=1 and k=10. Recall uses every evaluation query. Timing uses individual top-10 queries with one CPU thread, a seeded fixed subset of up to 1000 queries shared by every method, up to 50 warmup queries, and three repetitions. The reported mean is from the median repetition by mean latency. QPS=1000/mean milliseconds. Loading is excluded: FAISS reads the stored index into memory, and Vamana materializes its complete corpus as a contiguous float32 array before searching. Full timing observations are saved. Exact ground-truth generation may use batched calls; reported timing always uses individual queries.")
    if evidence["points"]:
        machine=evidence["points"][0]["machine"]
        paragraph(f"Timing host: {machine.get('host','unrecorded')}; CPU: {machine['cpu']}; logical CPUs: {machine['logical_cpus']}; benchmark threads: 1. Platform: {machine['platform']}. Python: {machine['python']}. Libraries: {machine['packages']}.")
        if machine.get("core_compiler"):
            paragraph(f"Core compiler: {machine['core_compiler']['compiler']}. Build flags: "+" ".join(x for x in machine['core_compiler']['command'] if x.startswith("-")))
    paragraph("Peak build memory is process high-water RSS measured in fresh workers, taking the maximum across Vamana attempts when resumed. FAISS index size is its serialized index including vectors. Vamana graph bytes are reported separately; its comparable searchable-index size adds the prepared float32 corpus file. PCA and source embeddings are excluded from searchable-index sizes. Construction times exclude dataset preparation, opening the prepared memory-mapped arrays, and final index serialization. They include any first-touch reads of mapped corpus pages during construction, so they are elapsed construction measurements rather than pure CPU-only kernel times. Vamana time also includes allocation, random initialization, entry selection, both construction passes, graph validation, checkpoint overhead, and recorded prior attempts. A forcibly killed attempt is recorded through its last heartbeat, so its total is explicitly a lower bound. Successful native-pass time is saved separately.")
    heading("Part 1 — Exact FAISS and HNSW")
    paragraph(f"HNSW construction uses M={config['hnsw']['M']} and efConstruction={config['hnsw']['efConstruction']}; efSearch varies over {config['hnsw']['efSearch']}. The build is single-threaded. Completed indices and exact neighbors are persisted.")
    for d in config["full_datasets"]:
        paragraph(d,"Heading3")
        selected=sorted([p for p in evidence["points"] if p["dataset"]==d and p["method"] in ("exact","hnsw")],key=lambda p:(p["method"],-1 if p["method"]=="exact" else int(p["search_parameter"])))
        metric_table(selected);image(f"{d}_hnsw.png")
        path=root/"truth"/f"{d}.json"
        if path.exists():paragraph(f"Exact-ground-truth verification: {read_json(path)}")
        if d=="sift1m":
            exact_points=[p for p in selected if p["method"]=="exact"]
            if exact_points:
                p=exact_points[0]
                paragraph(f"SIFT tie handling: exact FAISS achieves ID-based 1-recall@1={p['recall_1']:.5f} and 10-recall@10={p['recall_10']:.5f} against the supplied IDs. The saved verification compares the supplied neighbors' distances with exact FAISS distances; the sorted top-10 distances agree exactly in this run. Different IDs among equal-distance ties are evaluated literally against the supplied truth, so these exact-baseline recalls are not rounded up to 1.0.")
    heading("Part 2 — Vamana implementation and construction")
    paragraph("The original C++17 implementation uses fixed-width uint32 adjacency and degree arrays, contiguous float32 vectors, cached query distances, and generation-stamped distance/expansion arrays. Python invokes a C ABI through ctypes. There are no Python loops over vector coordinates. Candidate ordering breaks distance ties by row ID. GreedySearch returns the complete expanded-vertex set, including candidates later evicted. RobustPrune unions this set with current outgoing edges, excludes self, and uses alpha squared with squared Euclidean distances. Reverse-edge overflow invokes the same pruning routine. The entry is the point nearest the corpus mean, calculated in float64. Both passes visit independent seeded permutations: alpha=1 then configured alpha. Checkpoints preserve graph, current permutation, RNG state, and pass position. Builds are sequential to preserve these updates.")
    table(["Dataset","R / Lbuild / alpha","Build s","Degree mean / max","Graph MiB","Peak RSS MiB"],
        [[b["dataset"]," / ".join(str(b["spec"]["parameters"][k]) for k in ("R","L_build","alpha")),f"{b['build_seconds']:.2f}",
          f"{b['average_out_degree']:.2f} / {b['maximum_out_degree']}",f"{b['graph_bytes']/2**20:.2f}",f"{b['build_process_peak_rss_bytes']/2**20:.2f}"]
         for b in evidence["builds"].values() if b["method"]=="vamana" and b["dataset"] in config["full_datasets"]])
    paragraph(f"Build/search random seed: {config['seed']}. Default construction R=64, L_build=100, alpha=1.2. No automatic fallback changes these parameters. Every completed graph is checked for valid IDs, unique non-self outgoing edges, and degree <= R.")
    for build in evidence["builds"].values():
        for note in build.get("measurement_notes", []):
            paragraph(f"Measurement note for {build['dataset']}/{build['index_id']}: {note}")
    for b in evidence["builds"].values():
        if b["method"]=="vamana" and b.get("attempt_timing_lower_bound"):
            paragraph(f"Timing limitation for {b['dataset']}/{b['index_id']}: a prior process ended without a final timing heartbeat. Reported total construction time is a lower bound; inspect the attempt records and task logs.")
    scaling=root/"scaling.json"
    if scaling.exists():
        value=read_json(scaling)
        table(["Subset N","Construction s","Peak RSS MiB"],[[p["points"],f"{p['build_seconds']:.2f}",f"{p['peak_build_rss_bytes']/2**20:.2f}"] for p in value["measurements"]])
        paragraph(f"Scaling exponent: {value['empirical_exponent']:.3f}. Estimated full build seconds: {value['estimated_build_seconds']}. {value['limitation']}.")
    trace=root/"sift100k_trace.json"
    if trace.exists():
        value=read_json(trace)
        paragraph(f"SIFT100K query row {value['query_row']}: exact top-10={value['exact_top10_ids']}; Vamana top-10={value['ids']}; 10-recall@10={value['recall_10']:.3f}; expanded={value['expanded']}; distinct distance evaluations={value['distance_evaluations']}. IDs refer to the subset row space.")
    heading("Part 3a — SIFT100K parameter study")
    study=sorted([p for p in evidence["points"] if p["dataset"]=="sift100k"],key=lambda p:(p["index_id"],p["search_parameter"]))
    table(["Graph / operating point","R","alpha","L_build","L_search"],
          [[p["index_id"]+" / "+str(p["L_search"]),p["R"],p["alpha"],p["L_build"],p["L_search"]] for p in study])
    metric_table(study,counters=True);image("sift100k_search_list.png");image("sift100k_latency.png")
    paragraph("A larger L_search retains more alternatives, usually improving recall at additional distance and traversal cost. R changes the available routes and the cost per expansion. More outgoing edges can expose a useful distant region sooner and therefore reduce expansions at comparable recall, even while increasing distances per expansion. Increasing alpha makes pruning harder and retains more alternate, potentially long-range connections. These tendencies do not guarantee monotonic latency or recall in every measured run.")
    for name in sorted({p["index_id"] for p in study}):
        series=sorted([p for p in study if p["index_id"]==name],key=lambda p:p["L_search"])
        if len(series)>1:
            a,b=series[0],series[-1]
            paragraph(f"Measured {name}: L_search {a['L_search']}→{b['L_search']}, recall {a['recall_10']:.4f}→{b['recall_10']:.4f}, mean latency {a['mean_latency_ms']:.4f}→{b['mean_latency_ms']:.4f} ms, expansions/query {a['vertices_expanded_per_query']:.2f}→{b['vertices_expanded_per_query']:.2f}, distances/query {a['distance_evaluations_per_query']:.2f}→{b['distance_evaluations_per_query']:.2f}.")
    common_L=100 if 100 in config["vamana"]["L_search"] else config["vamana"]["L_search"][0]
    reference=next((p for p in study if p["L_search"]==common_L and all(p[k]==config["vamana"][k] for k in ("R","L_build","alpha"))),None)
    if reference:
        for params in config["study"]:
            if all(params[k]==config["vamana"][k] for k in ("R","L_build","alpha")):continue
            variant=next((p for p in study if p["L_search"]==common_L and all(p[k]==params[k] for k in ("R","L_build","alpha"))),None)
            if variant:
                changed=", ".join(f"{k}: {reference[k]} to {variant[k]}" for k in ("R","alpha","L_build") if reference[k]!=variant[k])
                paragraph(f"Controlled construction comparison at L_search={common_L} ({changed}): 10-recall@10 changes from {reference['recall_10']:.5f} to {variant['recall_10']:.5f}; mean latency from {reference['mean_latency_ms']:.4f} to {variant['mean_latency_ms']:.4f} ms; expansions/query from {reference['vertices_expanded_per_query']:.2f} to {variant['vertices_expanded_per_query']:.2f}; distinct distances/query from {reference['distance_evaluations_per_query']:.2f} to {variant['distance_evaluations_per_query']:.2f}. All other construction settings and the corpus/query sets are held fixed.")
    for target in (.8,.9,.95):
        summaries=[]
        for name in sorted({p["index_id"] for p in study}):
            candidates=[p for p in study if p["index_id"]==name and p["recall_10"]>=target]
            if candidates:
                best=min(candidates,key=lambda p:p["mean_latency_ms"])
                summaries.append(f"{name}: {best['mean_latency_ms']:.4f} ms, {best['vertices_expanded_per_query']:.2f} expansions, {best['distance_evaluations_per_query']:.2f} distances, achieved recall {best['recall_10']:.4f}")
        if summaries:paragraph(f"Fastest measured settings reaching recall >= {target:.2f}: "+"; ".join(summaries)+". Achieved recalls differ; these are threshold comparisons.")
    heading("Part 3b — Full-dataset comparisons")
    for d in config["full_datasets"]:
        paragraph(d,"Heading3");image(f"{d}_comparison.png")
        rows,gap=comparable(evidence["points"],d);metric_table(rows,counters=True)
        if gap is not None:
            h,v=rows[-2:]
            paragraph(f"The closest measured HNSW/Vamana recall pair has absolute 10-recall@10 gap {gap:.5f}. HNSW/Vamana latency ratio={h['mean_latency_ms']/v['mean_latency_ms']:.3f}; build-time ratio={h['build_seconds']/v['build_seconds']:.3f}; index-size ratio={h['index_bytes']/v['index_bytes']:.3f}. These ratios compare the displayed operating points, rather than interpolated equal-recall settings.")
            exact=next((p for p in rows if p["method"]=="exact"),None)
            if exact:
                paragraph(f"At the displayed settings, HNSW queries are {exact['mean_latency_ms']/h['mean_latency_ms']:.2f} times as fast as the exact scan and Vamana queries are {exact['mean_latency_ms']/v['mean_latency_ms']:.2f} times as fast. Relative to the exact index, searchable storage is {h['index_bytes']/exact['index_bytes']:.3f} times for HNSW and {v['index_bytes']/exact['index_bytes']:.3f} times for Vamana. Graph links trade extra storage and upfront construction for avoiding a complete corpus scan on every query.")
            paragraph("The Vamana builder performs two full corpus passes and reciprocal overflow pruning. Its search repeatedly sorts and deduplicates a bounded candidate vector, which adds implementation overhead beyond distance calculations. The measurements compare this original implementation with FAISS at the reported settings; they do not establish a universal performance ordering between the algorithms. No profiling measurement isolates the share of time spent in sorting or pruning. Vamana's fixed-width adjacency reserves R slots per vertex even when the measured degree is smaller, so a lower degree alone does not reduce its serialized allocation. The SIFT100K study measures the quality and search-work consequences of reducing R before changing that memory tradeoff.")
    heading("Sources and reproducibility")
    paragraph("ProblemStatement.pdf (CS6101 PA2, 2026); DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node, NeurIPS 2019, https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf; embeddings: https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3; SIFT1M: https://corpus-texmex.irisa.fr/. Scripts, configurations, compact measurements, and reproduction commands accompany this report.")
    heading("LLM usage documentation")
    paragraph("LLM assistance was used for planning, implementation, debugging, and validation. "+
              ("The available local transcript snapshots follow. Genuine shareable links and confirmation that every used chat is included are still required; see the missing-details list on the first page. Refresh active-session exports after the conversations finish. " if draft else
               "The supplied complete chat transcripts and shareable links follow, as required by the assignment. ")+
              "LLM documentation does not replace measured experimental evidence.")
    for number,chat in enumerate(details.get("llm_chats",[]),1):
        paragraph(f"Chat {number}: {chat.get('description','')}")
        url=chat.get("share_url","")
        if url:
            story.append(Paragraph(f'<link href="{html.escape(url,quote=True)}">{html.escape(url)}</link>',styles["BodyText"]))
        path=pathlib.Path(details_path).parent/chat.get("transcript_path","")
        if path.is_file():
            for line in path.read_text().splitlines():
                paragraph(line or " ")
                for attachment in transcript_attachments(line,path.parent):
                    try:
                        ImageReader(str(attachment)).getRGBData()
                    except (OSError,ValueError) as error:
                        paragraph(f"Image attachment {attachment.name} could not be displayed ({type(error).__name__}). Its original bytes and checksum are preserved in the accompanying transcript attachments.")
                        continue
                    picture=Image(str(attachment))
                    scale=min(6.5*inch/picture.imageWidth,8*inch/picture.imageHeight)
                    picture.drawWidth=picture.imageWidth*scale
                    picture.drawHeight=picture.imageHeight*scale
                    story.extend([picture,Spacer(1,8)])
    path=root/("report_draft.pdf" if draft else "report.pdf")
    doc=SimpleDocTemplate(str(path),pagesize=(8.5*inch,11*inch),leftMargin=.8*inch,rightMargin=.8*inch,topMargin=.65*inch,bottomMargin=.65*inch)
    def page_number(canvas,document):
        canvas.setFont("PA2Sans",8);canvas.drawRightString(7.7*inch,.35*inch,str(document.page))
    doc.build(story,onFirstPage=page_number,onLaterPages=page_number)
    flattened=[{k:v for k,v in p.items() if k not in ("machine","policy","parameters")} for p in evidence["points"]]
    save_csv(root/"metrics.csv",flattened)
    if not draft:
        write_json(root/"report_manifest.json",{"draft":False,"pdf_sha256":sha256(path),"details_sha256":sha256(details_path),
                    "evidence_fingerprint":fingerprint(evidence),"figure_sha256":{f.name:sha256(f) for f in figures},
                    "transcript_sha256":{chat["transcript_path"]:sha256(pathlib.Path(details_path).parent/chat["transcript_path"]) for chat in details["llm_chats"]},
                    "attachment_sha256":{p.relative_to(pathlib.Path(details_path).parent).as_posix():sha256(p) for chat in details["llm_chats"]
                        for p in transcript_attachments((pathlib.Path(details_path).parent/chat["transcript_path"]).read_text(),(pathlib.Path(details_path).parent/chat["transcript_path"]).parent)}})
    return path


def package(config,details_path,output_dir=None):
    summary,_=audit(config,details_path,include_report=True,verify_checksums=True)
    if not summary["complete"]:
        raise ValueError("Submission is incomplete: "+"; ".join(summary["missing_or_invalid"]))
    details,_=submission_details(details_path)
    out=pathlib.Path(output_dir) if output_dir else ROOT/"submission"
    out.mkdir(parents=True,exist_ok=True)
    archive=out/("_".join(details["roll_numbers"])+".zip")
    result_root=pathlib.Path(config["results_dir"])
    entries=[]
    for name in ["pa2","cpp","scripts","configs","cloud","tests"]:
        for path in (ROOT/name).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc",".partial") and path.name!="runtime.json":
                entries.append((path,path.relative_to(ROOT).as_posix()))
    for name in ["README.md","pyproject.toml"]:
        entries.append((ROOT/name,name))
    for path in result_root.rglob("*"):
        if path.is_file() and path.suffix in (".json",".csv",".png",".pdf",".txt") and "progress" not in path.parts and path.name!="report_draft.pdf":
            entries.append((path,"results/"+path.relative_to(result_root).as_posix()))
    # Include the exact validated config even when a non-default config was used.
    entries.append((pathlib.Path(config["config_path"]),"experiment_config.json"))
    for number,chat in enumerate(details["llm_chats"],1):
        transcript=pathlib.Path(details_path).parent/chat["transcript_path"]
        entries.append((transcript,f"llm_transcripts/chat_{number}.txt"))
        for attachment in transcript_attachments(transcript.read_text(),transcript.parent):
            entry=(attachment,"llm_transcripts/attachments/"+attachment.name)
            if entry not in entries:entries.append(entry)
    entries.append((pathlib.Path(details_path),"submission_details.json"))
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as output:
        for path,name in entries:
            output.write(path,name)
    with zipfile.ZipFile(archive) as output:
        if output.testzip() is not None:
            raise ValueError("submission archive failed integrity verification")
        forbidden=[n for n in output.namelist() if n.startswith(("data/","indices/",".venv/",".aws/",".git/")) or n.endswith((".fvecs",".ivecs",".f32",".faiss",".vamana",".checkpoint"))]
        if forbidden:
            raise ValueError(f"unexpected large/private archive entries: {forbidden}")
    return archive
