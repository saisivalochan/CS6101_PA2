import array
import math
import pathlib
import random
import tempfile
import unittest

from pa2.core import Vamana


def flatten(points):
    return array.array("f", (v for p in points for v in p))


def reference_search(points, rows, entry, query, k, L):
    def distance(i):
        return sum((a-b)**2 for a,b in zip(points[i],query))
    candidates, visited, seen = {entry}, [], {entry}
    while candidates.difference(visited):
        p = min(candidates.difference(visited),key=lambda i:(distance(i),i))
        visited.append(p)
        candidates.update(rows[p])
        seen.update(rows[p])
        candidates=set(sorted(candidates,key=lambda i:(distance(i),i))[:L])
    result=sorted(candidates,key=lambda i:(distance(i),i))[:k]
    return result,visited,len(seen)


def reference_prune(points, rows, p, ids, alpha, R):
    def distance(a,b):
        return math.sqrt(sum((x-y)**2 for x,y in zip(points[a],points[b])))
    candidates=(set(ids)|set(rows[p]))-{p}
    result=[]
    while candidates:
        star=min(candidates,key=lambda i:(distance(p,i),i))
        result.append(star);candidates.remove(star)
        if len(result)==R:
            break
        candidates={i for i in candidates if alpha*distance(star,i)>distance(p,i)}
    return result


class CoreTests(unittest.TestCase):
    def test_random_search_matches_literal_algorithm(self):
        rng=random.Random(27)
        points=[[rng.randrange(-10,11) for _ in range(4)] for _ in range(50)]
        rows=[rng.sample([j for j in range(50) if i!=j],6) for i in range(50)]
        with Vamana(flatten(points),6,4) as index:
            index.set_graph(rows,entry=17)
            for L in [3,5,10,30,50]:
                for _ in range(8):
                    q=[rng.randrange(-12,13) for _ in range(4)]
                    expected,visited,evaluations=reference_search(points,rows,17,q,3,L)
                    actual=index.search(array.array("f",q),3,L,True)
                    self.assertEqual(actual["ids"],expected)
                    self.assertEqual(actual["visited"],visited)
                    self.assertEqual(actual["distance_evaluations"],evaluations)
                    self.assertEqual(actual["expanded"],len(visited))

    def test_evicted_expansions_remain_visited_and_distances_are_cached(self):
        points=[[10],[9],[8],[7],[0]]
        rows=[[1,2],[0,4],[1,3],[2,4],[0,3]]
        with Vamana(flatten(points),2,1) as index:
            index.set_graph(rows)
            actual=index.search(array.array("f",[0]),1,1,True)
            expected,visited,count=reference_search(points,rows,0,[0],1,1)
            self.assertEqual(actual["ids"],expected)
            self.assertEqual(actual["visited"],visited)
            self.assertGreater(len(visited),1)
            self.assertIn(0,visited)
            self.assertEqual(actual["distance_evaluations"],count)
            self.assertLessEqual(count,len(points))

    def test_prune_uses_euclidean_alpha_with_squared_distances(self):
        points=[[0,0],[2,0],[2,4],[100,100]]
        with Vamana(flatten(points),2,2) as index:
            index.set_graph([[],[],[],[]])
            index.prune(0,[0,1,1,2],1.2)
            self.assertEqual(index.graph()[0][0],[1,2])
            index.prune(0,[1,2],1)
            self.assertEqual(index.graph()[0][0],[1])

    def test_random_prune_matches_literal_algorithm_and_unions_existing_edges(self):
        rng=random.Random(55)
        points=[[rng.randrange(-20,21) for _ in range(3)] for _ in range(25)]
        rows=[rng.sample([j for j in range(25) if j!=i],4) for i in range(25)]
        for alpha in [1,1.2,1.5,2]:
            with Vamana(flatten(points),4,3) as index:
                index.set_graph(rows)
                for p in range(25):
                    ids=rng.sample(range(25),15)
                    expected=reference_prune(points,rows,p,ids,alpha,4)
                    index.prune(p,ids,alpha)
                    self.assertEqual(index.graph()[0][p],expected)

    def test_equality_prunes_and_excludes_self(self):
        points=[[0],[0],[1],[2]]
        with Vamana(flatten(points),3,1) as index:
            index.set_graph([[],[],[],[]])
            index.prune(0,[0,1,2,3],1)
            self.assertEqual(index.graph()[0][0],[1])

    def test_build_is_deterministic_centroid_entry_and_persistence(self):
        rng=random.Random(10)
        points=[[rng.randrange(-9,10) for _ in range(5)] for _ in range(90)]
        mean=[sum(p[j] for p in points)/len(points) for j in range(5)]
        entry=min(range(len(points)),key=lambda i:sum((a-b)**2 for a,b in zip(points[i],mean)))
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)/"index.vamana"
            with Vamana(flatten(points),8,5) as first, Vamana(flatten(points),8,5) as second:
                first.build(20,1.2,42,progress=lambda *args:False)
                second.build(20,1.2,42,progress=lambda *args:False)
                rows,actual_entry=first.graph()
                self.assertEqual((rows,actual_entry),second.graph())
                self.assertEqual(actual_entry,entry)
                self.assertTrue(all(0<len(row)<=8 and len(row)==len(set(row)) and i not in row for i,row in enumerate(rows)))
                first.validate(); first.save(path)
                second.load(path)
                self.assertEqual(first.graph(),second.graph())
                query=array.array("f",points[0])
                self.assertEqual(first.search(query,10,90),second.search(query,10,90))

    def test_checkpoint_resume_is_identical_to_uninterrupted_build(self):
        rng=random.Random(42)
        points=[[rng.randrange(-30,31) for _ in range(3)] for _ in range(1100)]
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)/"checkpoint"
            with Vamana(flatten(points),5,3) as resumed, Vamana(flatten(points),5,3) as full:
                with self.assertRaisesRegex(RuntimeError,"interrupted"):
                    resumed.build(10,1.2,42,path,progress=lambda *args:True)
                resumed.load(path)
                resumed.build(10,1.2,42,path,progress=lambda *args:False)
                full.build(10,1.2,42,progress=lambda *args:False)
                self.assertEqual(resumed.graph(),full.graph())
                self.assertGreater(resumed.info()["build_seconds"],0)

    def test_invalid_parameters_and_corrupt_file(self):
        with self.assertRaises(RuntimeError):
            Vamana(array.array("f",[1,2]),2,1)
        with Vamana(array.array("f",[1,2,3]),1,1) as index:
            with self.assertRaises(ValueError):
                index.search(array.array("f",[1]),2,1)
            with self.assertRaises(RuntimeError):
                index.set_graph([[0],[2],[1]])
            with tempfile.TemporaryDirectory() as tmp:
                p=pathlib.Path(tmp)/"bad"
                p.write_bytes(b"garbage!")
                with self.assertRaises(RuntimeError):
                    index.load(p)


if __name__=="__main__":
    unittest.main()
