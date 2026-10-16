"""Small ctypes binding; the core also supports dependency-free Python arrays."""
import array
import ctypes as C
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
U32 = C.c_uint32
U64 = C.c_uint64
F32 = C.c_float
P32 = C.POINTER(U32)
PF = C.POINTER(F32)
PROGRESS = C.CFUNCTYPE(C.c_int, U32, U64, U64, C.c_double)
_LIB = None


def library():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = ROOT / "build/libvamana.so"
    source = ROOT / "cpp/vamana.cpp"
    if not path.exists() or path.stat().st_mtime < source.stat().st_mtime:
        subprocess.run([sys.executable, str(ROOT / "scripts/build_core.py")], check=True)
    lib = C.CDLL(str(path))
    lib.va_error.restype = C.c_char_p
    lib.va_create.argtypes = [PF, U64, U32, U32]
    lib.va_create.restype = C.c_void_p
    lib.va_free.argtypes = [C.c_void_p]
    lib.va_free.restype = None
    functions = {
        "va_set_graph": [C.c_void_p, P32, P32, U32],
        "va_get_graph": [C.c_void_p, P32, P32, P32],
        "va_prune": [C.c_void_p, U32, P32, U64, C.c_double],
        "va_search": [C.c_void_p, PF, U32, U32, P32, PF, P32, C.POINTER(U64)],
        "va_build": [C.c_void_p, U32, C.c_double, U64, C.c_char_p, C.c_double, PROGRESS],
        "va_save": [C.c_void_p, C.c_char_p], "va_load": [C.c_void_p, C.c_char_p],
        "va_validate": [C.c_void_p],
        "va_info": [C.c_void_p, P32, C.POINTER(C.c_double), P32, C.POINTER(C.c_double)],
    }
    for name, args in functions.items():
        function = getattr(lib, name)
        function.argtypes = args
        function.restype = C.c_int
    _LIB = lib
    return lib


def pointer(data, scalar):
    if isinstance(data, array.array):
        if data.itemsize != C.sizeof(scalar):
            raise ValueError("array element size mismatch")
        return C.cast(data.buffer_info()[0], C.POINTER(scalar))
    return data.ctypes.data_as(C.POINTER(scalar))


class Vamana:
    """One index and reusable output buffers. Calls on one instance are sequential."""
    def __init__(self, vectors, R, dim=None):
        self.lib = library()
        self.vectors = vectors  # Keeps native input storage alive.
        if isinstance(vectors, array.array):
            if vectors.typecode != "f" or not dim or len(vectors) % dim:
                raise ValueError("require float32 array with a valid dimension")
            self.n, self.dim = len(vectors) // dim, dim
        else:
            if vectors.ndim != 2 or vectors.dtype.name != "float32" or not vectors.flags.c_contiguous:
                raise ValueError("require C-contiguous float32 (N,D) array")
            self.n, self.dim = vectors.shape
        self.R = int(R)
        self.handle = self.lib.va_create(pointer(vectors, F32), self.n, self.dim, self.R)
        if not self.handle:
            raise RuntimeError(self.lib.va_error().decode())
        self._buffers = {}

    def _check(self, status):
        if status:
            raise RuntimeError(self.lib.va_error().decode())

    def close(self):
        if getattr(self, "handle", None):
            self.lib.va_free(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    def set_graph(self, rows, entry=0):
        if len(rows) != self.n or any(len(row) > self.R for row in rows):
            raise ValueError("graph shape mismatch")
        edges = array.array("I", [0]) * (self.n*self.R)
        degrees = array.array("I", map(len, rows))
        for p, row in enumerate(rows):
            edges[p*self.R:p*self.R+len(row)] = array.array("I", row)
        self._check(self.lib.va_set_graph(self.handle, pointer(edges, U32), pointer(degrees, U32), entry))

    def graph(self):
        edges = array.array("I", [0]) * (self.n*self.R)
        degrees = array.array("I", [0]) * self.n
        entry = U32()
        self._check(self.lib.va_get_graph(self.handle, pointer(edges,U32),pointer(degrees,U32),C.byref(entry)))
        return [list(edges[p*self.R:p*self.R+degrees[p]]) for p in range(self.n)], entry.value

    def prune(self, p, candidates, alpha):
        ids = array.array("I", candidates)
        self._check(self.lib.va_prune(self.handle, p, pointer(ids,U32),len(ids),alpha))

    def search_raw(self, query, k=10, L=100, include_visited=False):
        if not self.handle:
            raise RuntimeError("index is closed")
        if isinstance(query,array.array):
            if query.typecode != "f" or len(query)!=self.dim:
                raise ValueError("query shape/type mismatch")
        elif query.shape != (self.dim,) or query.dtype.name != "float32" or not query.flags.c_contiguous:
            raise ValueError("require contiguous float32 query of shape (D,)")
        if not 0 < k <= self.n or L < k:
            raise ValueError("require 0<k<=N and L>=k")
        if k not in self._buffers:
            self._buffers[k] = ((U32*k)(),(F32*k)(),(U64*2)())
        ids, distances, stats = self._buffers[k]
        visited = (U32*self.n)() if include_visited else None
        self._check(self.lib.va_search(self.handle,pointer(query,F32),k,L,ids,distances,visited,stats))
        return ids, distances, stats, visited

    def search(self, query, k=10, L=100, include_visited=False):
        ids, distances, stats, visited = self.search_raw(query,k,L,include_visited)
        result = {"ids": [int(x) if x != 2**32-1 else -1 for x in ids], "distances":list(distances),
                  "expanded":int(stats[0]),"distance_evaluations":int(stats[1])}
        if include_visited:
            result["visited"] = list(visited[:stats[0]])
        return result

    def build(self, L_build=100, alpha=1.2, seed=42, checkpoint=None, checkpoint_seconds=900, progress=None):
        errors = []
        def callback(pass_number, processed, total, seconds):
            try:
                if progress:
                    return int(bool(progress(pass_number,processed,total,seconds)))
                print(json.dumps({"pass":pass_number,"processed":processed,"total":total,"build_seconds":seconds}),flush=True)
                return 0
            except BaseException as error:
                errors.append(error)
                return 1
        handler = PROGRESS(callback)
        status = self.lib.va_build(self.handle,L_build,alpha,seed,
                                  str(checkpoint).encode() if checkpoint else None,
                                  checkpoint_seconds,handler)
        if errors:
            raise errors[0]
        self._check(status)

    def save(self, path):
        pathlib.Path(path).parent.mkdir(parents=True,exist_ok=True)
        self._check(self.lib.va_save(self.handle,str(path).encode()))

    def load(self, path):
        self._check(self.lib.va_load(self.handle,str(path).encode()))
        return self

    def validate(self):
        self._check(self.lib.va_validate(self.handle))

    def info(self):
        entry, maximum = U32(), U32()
        average, elapsed = C.c_double(), C.c_double()
        self._check(self.lib.va_info(self.handle,C.byref(entry),C.byref(average),C.byref(maximum),C.byref(elapsed)))
        return {"entry":entry.value,"average_out_degree":average.value,"maximum_out_degree":maximum.value,"build_seconds":elapsed.value}
