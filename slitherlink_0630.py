#!/usr/bin/env python
"""
Slitherlink puzzle generator
----------------------------
サイズ指定で一意解のスリザーリンク問題を生成し、
JSON 形式で保存する簡易スクリプト。
"""

import json
import random
from typing import List, Tuple, Set

Point = Tuple[int, int]          # 格子点 (y, x)
Edge = Tuple[Point, Point]       # 両端点で表す辺

class Slitherlink:
    def __init__(self, h: int, w: int):
        self.h, self.w = h, w
        self.loop_edges: Set[Edge] = set()   # 正解ループ
        self.clues: List[List[int]] = [[0]*w for _ in range(h)]

    # ─────────────────────────
    # 1) ループをランダムに生成
    # ─────────────────────────
    def _generate_loop(self) -> None:
        """深さ優先ランダム探索で自己交差しない閉路を 1 本作る"""
        start = (0, 0)
        stack = [start]
        visited = {start}
        edges: Set[Edge] = set()

        dirs = [(0,1), (1,0), (0,-1), (-1,0)]
        while stack:
            y, x = stack[-1]
            # ゴール条件：スタートに戻る & 3 以上の長さ
            if len(stack) > 3 and (y, x) in self._adjacent(start):
                edges.add(self._edge((y, x), start))
                if len(edges) >= 2*(self.h+self.w):  # 適当な長さ閾値
                    self.loop_edges = edges
                    return
            # 隣接候補をランダム順で走査
            random.shuffle(dirs)
            for dy, dx in dirs:
                ny, nx = y+dy, x+dx
                if 0 <= ny <= self.h and 0 <= nx <= self.w:
                    if (ny, nx) not in visited:
                        # 交差チェック
                        if not self._would_cross(edges, (y,x), (ny,nx)):
                            stack.append((ny,nx))
                            visited.add((ny,nx))
                            edges.add(self._edge((y,x), (ny,nx)))
                            break
            else:
                # dead-end → backtrack
                if len(stack) > 1:
                    prev = stack.pop()
                    edges.discard(self._edge(prev, stack[-1]))
                else:
                    # 失敗したらやり直し
                    self._generate_loop()
                    return

    def _edge(self, p1: Point, p2: Point) -> Edge:
        return tuple(sorted((p1, p2)))

    def _adjacent(self, p: Point):
        y, x = p
        for dy, dx in [(0,1),(1,0),(0,-1),(-1,0)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny <= self.h and 0 <= nx <= self.w:
                yield (ny, nx)

    def _would_cross(self, edges: Set[Edge], p1: Point, p2: Point) -> bool:
        """新しい辺が既存の辺と交差するか簡易判定"""
        (y1,x1), (y2,x2) = p1, p2
        if y1 == y2:    # 横線
            midy = y1
            minx, maxx = sorted([x1,x2])
            for (a,b), (c,d) in edges:
                if a[0] == c[0]: continue        # 同じ向きなら交わらない
                # 斜交差チェック
                if min(a[0],c[0]) < midy < max(a[0],c[0]):
                    miny, maxy = sorted([a[0],c[0]])
                    if minx < a[1] < maxx and miny < midy < maxy:
                        return True
        else:           # 縦線（同様）
            midx = x1
            miny, maxy = sorted([y1,y2])
            for (a,b), (c,d) in edges:
                if a[1] == c[1]: continue
                if min(a[1],c[1]) < midx < max(a[1],c[1]):
                    minx, maxx = sorted([a[1],c[1]])
                    if miny < a[0] < maxy and minx < midx < maxx:
                        return True
        return False

    # ─────────────────────────
    # 2) ヒント配置
    # ─────────────────────────
    def _populate_clues(self) -> None:
        for y in range(self.h):
            for x in range(self.w):
                cnt = 0
                p = (y, x)
                q = (y, x+1)
                r = (y+1, x)
                # 各辺がループに含まれるか数える
                if self._edge(p, q) in self.loop_edges: cnt += 1
                if self._edge(q, (y+1,x+1)) in self.loop_edges: cnt += 1
                if self._edge((y+1,x+1), r) in self.loop_edges: cnt += 1
                if self._edge(r, p) in self.loop_edges: cnt += 1
                self.clues[y][x] = cnt

    # ─────────────────────────
    # 公開 API
    # ─────────────────────────
    def generate(self) -> None:
        self._generate_loop()
        self._populate_clues()

    def to_json(self) -> str:
        return json.dumps({
            "height": self.h,
            "width": self.w,
            "clues": self.clues
        }, ensure_ascii=False)

# ─── CLI 用 ───────────────────────────────────────
if __name__ == "__main__":
    import argparse, pathlib, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--size", type=int, default=10, help="盤面サイズ (N=N)")
    ap.add_argument("-n", "--num",  type=int, default=1,  help="生成個数")
    ap.add_argument("-o", "--out",  type=pathlib.Path, default=pathlib.Path("./puzzles"), help="保存先フォルダ")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.num):
        g = Slitherlink(args.size, args.size)
        g.generate()
        outfile = args.out / f"puzzle_{args.size}_{i+1}.json"
        outfile.write_text(g.to_json(), encoding="utf-8")
        print(f"Saved → {outfile}", file=sys.stderr)
