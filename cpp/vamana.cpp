// Original in-memory Vamana implementation; algorithms follow PA2 pages 3–4.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Item = std::pair<float, uint32_t>;
thread_local std::string last_error;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

struct SearchResult {
    std::vector<Item> candidates;
    std::vector<uint32_t> visited;
    uint64_t evaluations = 0;
};

struct Index {
    const float* points;
    uint64_t n;
    uint32_t dim, R, entry = 0;
    std::vector<uint32_t> edges, degree;
    // Generation tags avoid clearing O(N) structures before every query.
    std::vector<uint32_t> distance_tag, expanded_tag;
    std::vector<float> distance_cache;
    uint32_t generation = 0;
    bool build_initialized = false;
    uint32_t L_build = 0, pass = 0;
    uint64_t seed = 0, position = 0;
    double alpha = 1, elapsed = 0;
    std::mt19937_64 rng;
    std::vector<uint32_t> order;

    Index(const float* p, uint64_t count, uint32_t d, uint32_t r)
        : points(p), n(count), dim(d), R(r) {
        require(p && count > 1 && count <= std::numeric_limits<uint32_t>::max(), "invalid point count");
        require(d > 0 && r > 0 && r < count, "require dimension>0 and 0<R<N");
        require(count <= std::numeric_limits<size_t>::max() / r, "graph allocation overflow");
        edges.resize(static_cast<size_t>(n) * R);
        degree.resize(n);
        distance_tag.resize(n);
        expanded_tag.resize(n);
        distance_cache.resize(n);
    }
    const float* point(uint32_t id) const { return points + static_cast<size_t>(id) * dim; }
    float d2(const float* a, const float* b) const {
        float total = 0;
        // Compiled contiguous loop: no per-coordinate Python calls or allocation.
        for (uint32_t j = 0; j < dim; ++j) { const float v = a[j] - b[j]; total += v*v; }
        return total;
    }
    void next_generation() {
        if (++generation == 0) {
            std::fill(distance_tag.begin(), distance_tag.end(), 0);
            std::fill(expanded_tag.begin(), expanded_tag.end(), 0);
            generation = 1;
        }
    }
    SearchResult search(const float* q, uint32_t k, uint32_t L) {
        require(q && k > 0 && k <= n && L >= k, "require 0<k<=N and L>=k");
        L = static_cast<uint32_t>(std::min<uint64_t>(L, n));
        next_generation();
        SearchResult result;
        result.candidates.reserve(static_cast<size_t>(L) + R);
        result.visited.reserve(L);
        auto distance = [&](uint32_t id) {
            if (distance_tag[id] != generation) {
                distance_cache[id] = d2(q, point(id));
                require(std::isfinite(distance_cache[id]), "nonfinite distance");
                distance_tag[id] = generation;
                ++result.evaluations;
            }
            return distance_cache[id];
        };
        result.candidates.emplace_back(distance(entry), entry);
        while (true) {
            auto it = std::find_if(result.candidates.begin(), result.candidates.end(),
                [&](const Item& x) { return expanded_tag[x.second] != generation; });
            if (it == result.candidates.end()) break;
            const uint32_t p = it->second;
            expanded_tag[p] = generation;
            result.visited.push_back(p);
            const size_t start = static_cast<size_t>(p) * R;
            for (uint32_t j = 0; j < degree[p]; ++j) {
                const uint32_t neighbor = edges[start+j];
                result.candidates.emplace_back(distance(neighbor), neighbor);
            }
            std::sort(result.candidates.begin(), result.candidates.end());
            // Cached distances are identical for duplicate IDs, so pairs deduplicate.
            result.candidates.erase(std::unique(result.candidates.begin(), result.candidates.end()), result.candidates.end());
            if (result.candidates.size() > L) result.candidates.resize(L);
        }
        return result;
    }
    std::vector<uint32_t> neighbors(uint32_t p) const {
        const auto begin = edges.begin() + static_cast<size_t>(p)*R;
        return {begin, begin+degree[p]};
    }
    void set_neighbors(uint32_t p, const std::vector<uint32_t>& ids) {
        require(p < n && ids.size() <= R, "invalid adjacency length");
        degree[p] = static_cast<uint32_t>(ids.size());
        std::copy(ids.begin(), ids.end(), edges.begin()+static_cast<size_t>(p)*R);
        std::fill(edges.begin()+static_cast<size_t>(p)*R+ids.size(), edges.begin()+(static_cast<size_t>(p)+1)*R, 0);
    }
    void prune(uint32_t p, std::vector<uint32_t> ids, double a) {
        require(p < n && std::isfinite(a) && a >= 1, "invalid prune parameters");
        const auto old = neighbors(p);
        ids.insert(ids.end(), old.begin(), old.end());
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        std::vector<Item> candidates;
        candidates.reserve(ids.size());
        for (uint32_t id : ids) {
            require(id < n, "invalid prune candidate ID");
            if (id != p) {
                const float distance = d2(point(p),point(id));
                require(std::isfinite(distance), "nonfinite prune distance");
                candidates.emplace_back(distance, id);
            }
        }
        std::sort(candidates.begin(), candidates.end());
        std::vector<uint8_t> removed(candidates.size(), 0);
        std::vector<uint32_t> kept;
        kept.reserve(R);
        const double alpha_squared = a*a;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (removed[i]) continue;
            const uint32_t star = candidates[i].second;
            kept.push_back(star);
            if (kept.size() == R) break;
            for (size_t j = i+1; j < candidates.size(); ++j) {
                if (!removed[j] && alpha_squared * d2(point(star),point(candidates[j].second)) <= candidates[j].first)
                    removed[j] = 1;
            }
        }
        set_neighbors(p, kept);
    }
    void validate() const {
        require(entry < n, "entry out of range");
        std::vector<uint32_t> ids;
        ids.reserve(R);
        for (uint64_t p = 0; p < n; ++p) {
            require(degree[p] <= R, "degree exceeds R");
            ids.clear();
            const size_t start = static_cast<size_t>(p)*R;
            for (uint32_t j = 0; j < degree[p]; ++j) {
                const uint32_t id = edges[start+j];
                require(id < n && id != p, "invalid/self edge");
                ids.push_back(id);
            }
            std::sort(ids.begin(),ids.end());
            require(std::adjacent_find(ids.begin(),ids.end())==ids.end(), "duplicate edge");
        }
    }
    template<class T> static void write(std::ofstream& f, const T& x) {
        f.write(reinterpret_cast<const char*>(&x), sizeof(T));
    }
    template<class T> static void read(std::ifstream& f, T& x) {
        f.read(reinterpret_cast<char*>(&x), sizeof(T));
        require(bool(f), "truncated index file");
    }
    void save(const std::string& path) const {
        const uint32_t endian = 0x01020304;
        const std::string tmp = path + ".partial";
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        require(bool(f), "cannot open index output");
        f.write("VAMANA02",8);
        write(f,endian); write(f,n); write(f,dim); write(f,R); write(f,entry);
        const uint32_t initialized = build_initialized ? 1 : 0;
        write(f,initialized); write(f,L_build); write(f,alpha); write(f,seed);
        write(f,pass); write(f,position); write(f,elapsed);
        f.write(reinterpret_cast<const char*>(degree.data()),degree.size()*sizeof(uint32_t));
        f.write(reinterpret_cast<const char*>(edges.data()),edges.size()*sizeof(uint32_t));
        std::ostringstream rs; rs << rng;
        const auto state = rs.str(); const uint64_t length = state.size();
        write(f,length); f.write(state.data(),length);
        const uint64_t count = order.size(); write(f,count);
        f.write(reinterpret_cast<const char*>(order.data()),count*sizeof(uint32_t));
        f.flush(); require(bool(f), "index write failed"); f.close();
        require(std::rename(tmp.c_str(),path.c_str())==0,"atomic index replacement failed");
    }
    void load(const std::string& path) {
        std::ifstream f(path,std::ios::binary);
        require(bool(f), "cannot open index file");
        char magic[8] = {}; f.read(magic,8);
        require(bool(f) && std::string(magic,8)=="VAMANA02", "unsupported index format");
        uint32_t endian, d, r; uint64_t count;
        read(f,endian); read(f,count); read(f,d); read(f,r);
        require(endian==0x01020304 && count==n && d==dim && r==R,"index shape/byte order mismatch");
        read(f,entry); uint32_t initialized; read(f,initialized); build_initialized=initialized!=0;
        read(f,L_build); read(f,alpha); read(f,seed); read(f,pass); read(f,position); read(f,elapsed);
        require(pass<=2 && position<=n && std::isfinite(alpha) && alpha>=1 && std::isfinite(elapsed) && elapsed>=0,"invalid build state");
        f.read(reinterpret_cast<char*>(degree.data()),degree.size()*sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(edges.data()),edges.size()*sizeof(uint32_t));
        require(bool(f),"truncated graph");
        uint64_t length; read(f,length); require(length<=1000000,"invalid RNG state length");
        std::string state(length,' '); f.read(state.data(),length);
        std::istringstream rs(state); rs >> rng; require(bool(rs),"invalid RNG state");
        read(f,count); require(count==0 || count==n,"invalid permutation length");
        order.resize(count); f.read(reinterpret_cast<char*>(order.data()),count*sizeof(uint32_t));
        require(bool(f) && f.peek()==std::ifstream::traits_type::eof(),"truncated or trailing index data");
        if (build_initialized && pass<2) {
            require(order.size()==n && L_build>0,"missing build permutation");
            auto check=order; std::sort(check.begin(),check.end());
            for (uint64_t i=0;i<n;++i) require(check[i]==i,"invalid build permutation");
        }
        validate();
    }
    void initialize(uint32_t L, double a, uint64_t random_seed) {
        L_build=L; alpha=a; seed=random_seed; rng.seed(seed); pass=0;position=0;elapsed=0;
        std::uniform_int_distribution<uint32_t> choose(0,static_cast<uint32_t>(n-1));
        std::vector<uint32_t> row; row.reserve(R);
        for (uint64_t p=0;p<n;++p) {
            row.clear();
            while(row.size()<R) {
                const uint32_t id=choose(rng);
                if(id!=p && std::find(row.begin(),row.end(),id)==row.end()) row.push_back(id);
            }
            set_neighbors(static_cast<uint32_t>(p),row);
        }
        std::vector<double> mean(dim,0);
        for(uint64_t p=0;p<n;++p) for(uint32_t j=0;j<dim;++j) mean[j]+=points[p*dim+j];
        for(double& x:mean) x/=static_cast<double>(n);
        double best=std::numeric_limits<double>::infinity();
        for(uint64_t p=0;p<n;++p) {
            double distance=0;
            for(uint32_t j=0;j<dim;++j) {const double v=points[p*dim+j]-mean[j];distance+=v*v;}
            if(distance<best){best=distance;entry=static_cast<uint32_t>(p);}
        }
        require(std::isfinite(best),"nonfinite input vectors");
        order.resize(n);std::iota(order.begin(),order.end(),uint32_t{0});
        std::shuffle(order.begin(),order.end(),rng);build_initialized=true;
    }
    using Progress = int (*)(uint32_t,uint64_t,uint64_t,double);
    void build(uint32_t L,double a,uint64_t random_seed,const std::string& checkpoint,double checkpoint_seconds,Progress callback) {
        require(L>0 && a>=1 && std::isfinite(a) && checkpoint_seconds>=0,"invalid build parameters");
        const auto start=Clock::now();
        const double previous=elapsed;
        if(!build_initialized) initialize(L,a,random_seed);
        else require(L==L_build && a==alpha && random_seed==seed,"checkpoint configuration mismatch");
        auto elapsed_now=[&](){return previous+std::chrono::duration<double>(Clock::now()-start).count();};
        auto last_checkpoint=Clock::now();
        while(pass<2) {
            const double alpha_pass=pass==0 ? 1.0 : alpha;
            for(;position<n;) {
                const uint32_t p=order[position];
                auto found=search(point(p),1,L_build);
                prune(p,std::move(found.visited),alpha_pass);
                const auto outgoing=neighbors(p);
                for(uint32_t j:outgoing) {
                    auto candidates=neighbors(j);
                    if(std::find(candidates.begin(),candidates.end(),p)!=candidates.end()) continue;
                    candidates.push_back(p);
                    if(candidates.size()>R) prune(j,std::move(candidates),alpha_pass);
                    else set_neighbors(j,candidates);
                }
                ++position;
                if(position%1000==0 || position==n) {
                    elapsed=elapsed_now();
                    if(callback && callback(pass+1,position,n,elapsed)!=0) {
                        if(!checkpoint.empty()) save(checkpoint);
                        throw std::runtime_error("build interrupted; checkpoint saved");
                    }
                    if(!checkpoint.empty() && checkpoint_seconds>0 && std::chrono::duration<double>(Clock::now()-last_checkpoint).count()>=checkpoint_seconds) {
                        save(checkpoint);last_checkpoint=Clock::now();
                    }
                }
            }
            ++pass;position=0;
            if(pass<2) std::shuffle(order.begin(),order.end(),rng);
            else order.clear();
            elapsed=elapsed_now();
            if(!checkpoint.empty()) save(checkpoint);
        }
        elapsed=elapsed_now();validate();
    }
};
template<class F> int protect(F&& f) {
    try {last_error.clear(); f(); return 0;}
    catch(const std::exception& e){last_error=e.what();return -1;}
    catch(...){last_error="unknown C++ error";return -1;}
}
Index& index(void* handle){require(handle,"null index handle");return *static_cast<Index*>(handle);}
}

extern "C" {
const char* va_error(){return last_error.c_str();}
void* va_create(const float* points,uint64_t n,uint32_t dim,uint32_t R){
    try {last_error.clear();return new Index(points,n,dim,R);}
    catch(const std::exception& e){last_error=e.what();return nullptr;}
}
void va_free(void* handle){delete static_cast<Index*>(handle);}
int va_set_graph(void* h,const uint32_t* edges,const uint32_t* degrees,uint32_t entry){return protect([&]{
    auto& x=index(h);require(edges && degrees,"null graph arrays");
    require(entry<x.n,"entry out of range");
    std::vector<uint32_t> row;
    for(uint64_t p=0;p<x.n;++p){
        require(degrees[p]<=x.R,"degree exceeds R");row.clear();
        for(uint32_t j=0;j<degrees[p];++j){
            const uint32_t id=edges[p*x.R+j];require(id<x.n && id!=p,"invalid/self edge");row.push_back(id);
        }
        std::sort(row.begin(),row.end());require(std::adjacent_find(row.begin(),row.end())==row.end(),"duplicate edge");
    }
    std::copy(edges,edges+x.edges.size(),x.edges.begin());std::copy(degrees,degrees+x.n,x.degree.begin());x.entry=entry;x.validate();
});}
int va_get_graph(void* h,uint32_t* edges,uint32_t* degrees,uint32_t* entry){return protect([&]{
    auto& x=index(h);if(edges)std::copy(x.edges.begin(),x.edges.end(),edges);if(degrees)std::copy(x.degree.begin(),x.degree.end(),degrees);if(entry)*entry=x.entry;
});}
int va_prune(void* h,uint32_t p,const uint32_t* candidates,uint64_t count,double alpha){return protect([&]{
    require(candidates || count==0,"null candidates");std::vector<uint32_t> ids;if(count)ids.assign(candidates,candidates+count);index(h).prune(p,std::move(ids),alpha);
});}
int va_search(void* h,const float* q,uint32_t k,uint32_t L,uint32_t* ids,float* distances,uint32_t* visited,uint64_t* stats){return protect([&]{
    require(ids && distances && stats,"null output arrays");auto result=index(h).search(q,k,L);
    for(uint32_t i=0;i<k;++i){ids[i]=i<result.candidates.size()?result.candidates[i].second:UINT32_MAX;distances[i]=i<result.candidates.size()?result.candidates[i].first:std::numeric_limits<float>::infinity();}
    if(visited)std::copy(result.visited.begin(),result.visited.end(),visited);
    stats[0]=result.visited.size();stats[1]=result.evaluations;
});}
int va_build(void* h,uint32_t L,double alpha,uint64_t seed,const char* checkpoint,double seconds,Index::Progress callback){return protect([&]{
    index(h).build(L,alpha,seed,checkpoint?checkpoint:"",seconds,callback);
});}
int va_save(void* h,const char* path){return protect([&]{require(path,"null path");index(h).save(path);});}
int va_load(void* h,const char* path){return protect([&]{
    require(path,"null path");auto& x=index(h);Index loaded(x.points,x.n,x.dim,x.R);
    loaded.load(path);std::swap(x,loaded);
});}
int va_validate(void* h){return protect([&]{index(h).validate();});}
int va_info(void* h,uint32_t* entry,double* average,uint32_t* maximum,double* elapsed){return protect([&]{
    auto& x=index(h);if(entry)*entry=x.entry;
    if(average)*average=std::accumulate(x.degree.begin(),x.degree.end(),uint64_t{0})/static_cast<double>(x.n);
    if(maximum)*maximum=*std::max_element(x.degree.begin(),x.degree.end());
    if(elapsed)*elapsed=x.elapsed;
});}
}
