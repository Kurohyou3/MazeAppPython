# coding: utf-8
"""
graph_maze.ipynb の処理を Python スクリプト化したもの。
遠隔環境で接続が切れても処理を続けられるよう、
nohup 等で実行することを想定しています。
実行例:
    nohup python3 run_graph_maze.py -n 10 -w 4 > log.txt 2>&1 &

オプションで生成個数(-n)や並列実行数(-w)を指定できます。
生成される迷路 JSON は sato/map_data フォルダに保存されます。
ノートブック版と同じ内容を出力します。
"""

import os
import json
import random
import sys
from pathlib import Path  # パス操作を簡潔にするために利用
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# -------------------- 調整可能なパラメータ --------------------
# スクリプトと同じディレクトリに map_data フォルダを作成
DEFAULT_OUT_DIR = Path(__file__).parent / "map_data"

@dataclass
class Config:
    """迷路生成に必要なパラメータをまとめた設定クラス"""
    size: int = 10                 # 盤面の幅＝高さ
    max_straight: int = 8          # 連続直線通路の最大長
    max_branch_dist: int = 8       # 分岐（次数≠2）間の最長距離
    loop_rate: float = 0.4         # ループ付与確率
    max_diameter: int = 30         # グラフ径の上限
    maze_count: int = 1            # 生成する迷路数
    seed: int | None = None        # 乱数シード（None で毎回ランダム）
    max_attempts_per_maze: int = 500  # 径が条件に合うまでの試行上限
    # 出力ディレクトリの Path オブジェクト
    out_dir: Path = field(default_factory=lambda: DEFAULT_OUT_DIR)

# グローバル設定インスタンス
config = Config()

# -------------------- データ構造 --------------------
Cell = Tuple[int, int]
Adj  = Dict[Cell, List[Cell]]

@dataclass
class Maze:
    """迷路を表す壁情報"""
    v: List[List[bool]]  # 縦壁 (y,x) True=壁あり (x,y)-(x+1,y) 間
    h: List[List[bool]]  # 横壁 (y,x) True=壁あり (x,y)-(x,y+1) 間

# -------------------- 基本ユーティリティ --------------------
def neigh(x: int, y: int):
    """周囲 4 方向の座標を返すジェネレータ"""
    if x > 0:
        yield 'L', x-1, y
    if x < config.size-1:
        yield 'R', x+1, y
    if y > 0:
        yield 'U', x, y-1
    if y < config.size-1:
        yield 'D', x, y+1

def carve(m: Maze, x: int, y: int, d: str):
    """指定方向の壁を取り除く"""
    if d == 'L':
        m.v[y][x-1] = False
    elif d == 'R':
        m.v[y][x] = False
    elif d == 'U':
        m.h[y-1][x] = False
    else:  # 'D'
        m.h[y][x] = False

def wall(m: Maze, x: int, y: int, d: str) -> bool:
    """その方向に壁があるかを判定"""
    return (
        m.v[y][x-1] if d == 'L' else
        m.v[y][x]   if d == 'R' else
        m.h[y-1][x] if d == 'U' else
        m.h[y][x]
    )

# -------------------- 2×2 通路チェック --------------------
def makes_open_square(m: Maze, x: int, y: int, d: str) -> bool:
    blocks = []
    if d in ('L', 'R'):
        dx = -1 if d == 'L' else 0
        for dy in (0, -1):
            blocks.append((x+dx, y+dy))
    else:
        dy = -1 if d == 'U' else 0
        for dx in (0, -1):
            blocks.append((x+dx, y+dy))

    for cx, cy in blocks:
        if 0 <= cx < config.size-1 and 0 <= cy < config.size-1:
            v0 = not m.v[cy][cx];      v1 = not m.v[cy+1][cx]
            h0 = not m.h[cy][cx];      h1 = not m.h[cy][cx+1]
            # 仮に掘った場合の壁状態を反映
            if d == 'L' and cx == x-1 and cy in (y-1, y):   v0 = True
            if d == 'R' and cx == x   and cy in (y-1, y):   v1 = True
            if d == 'U' and cy == y-1 and cx in (x-1, x):   h0 = True
            if d == 'D' and cy == y   and cx in (x-1, x):   h1 = True
            if v0 and v1 and h0 and h1:
                return True
    return False

# -------------------- 仮掘削後の最大直線長 --------------------
def max_run_after_carve(m: Maze, x: int, y: int, d: str) -> int:
    def run_h(px: int, py: int) -> int:
        run = 1
        ix = px-1
        while ix >= 0 and not m.v[py][ix]:
            run += 1
            ix -= 1
        ix = px
        while ix < config.size-1 and not m.v[py][ix]:
            run += 1
            ix += 1
        return run

    def run_v(px: int, py: int) -> int:
        run = 1
        iy = py-1
        while iy >= 0 and not m.h[iy][px]:
            run += 1
            iy -= 1
        iy = py
        while iy < config.size-1 and not m.h[iy][px]:
            run += 1
            iy += 1
        return run

    cells = [(x, y)]
    if d == 'R':
        cells.append((x+1, y))
    elif d == 'L':
        cells.append((x-1, y))
    elif d == 'D':
        cells.append((x, y+1))
    else:
        cells.append((x, y-1))

    return max(max(run_h(cx, cy), run_v(cx, cy)) for cx, cy in cells)

# -------------------- 迷路の可視化 --------------------
def render_maze(m: Maze) -> str:
    rows: List[str] = []
    rows.append(" " + "_" * (config.size * 2 - 1))
    for y in range(config.size):
        line = ["|"]
        for x in range(config.size):
            south = m.h[y][x] if y < config.size-1 else True
            east  = m.v[y][x] if x < config.size-1 else True
            line.append("_" if south else " ")
            line.append("|" if east  else " ")
        rows.append("".join(line))
    return "\n".join(rows)

# -------------------- DFS 木生成 --------------------
def generate_tree() -> Maze:
    v = [[True]*(config.size-1) for _ in range(config.size)]
    h = [[True]*config.size     for _ in range(config.size-1)]
    # Maze オブジェクトを最初に作成しておく
    # dataclass を使うことで v と h のリストをまとめて管理できる
    maze = Maze(v, h)
    visited = [[False]*config.size for _ in range(config.size)]
    degree  = [[0]*config.size for _ in range(config.size)]

    sx, sy = random.randrange(config.size), random.randrange(config.size)
    stack = [(sx, sy, '', 0, 0)]  # (x, y, prev_dir, run_len, dist_from_branch)
    visited[sy][sx] = True

    while stack:
        x, y, pd, run, dist = stack[-1]
        cand = [(d, nx, ny) for d, nx, ny in neigh(x, y) if not visited[ny][nx]]

        if run >= config.max_straight:
            cand = [t for t in cand if t[0] != pd]
        if dist >= config.max_branch_dist and pd:
            cand = [t for t in cand if len(cand) >= 2]

        cand = [t for t in cand
                if not makes_open_square(maze, x, y, t[0])
                and max_run_after_carve(maze, x, y, t[0]) <= config.max_straight]

        if cand:
            d, nx, ny = random.choice(cand)
            carve(maze, x, y, d)
            degree[y][x]   += 1
            degree[ny][nx] += 1
            visited[ny][nx] = True
            nxt_dist = 0 if degree[ny][nx] >= 3 else dist + 1
            nxt_run  = run + 1 if d == pd else 1
            stack.append((nx, ny, d, nxt_run, nxt_dist))
        else:
            stack.pop()
    # 生成した Maze インスタンスをそのまま返す
    return maze

# -------------------- ループ追加 --------------------
def add_loops(m: Maze):
    walls = (
        [(x, y, 'R') for y in range(config.size)   for x in range(config.size-1) if m.v[y][x]] +
        [(x, y, 'D') for y in range(config.size-1) for x in range(config.size)   if m.h[y][x]]
    )
    random.shuffle(walls)
    for x, y, d in walls:
        if random.random() > config.loop_rate:
            continue
        if makes_open_square(m, x, y, d):
            continue
        if max_run_after_carve(m, x, y, d) > config.max_straight:
            continue
        carve(m, x, y, d)

# -------------------- 後検証 --------------------
def is_connected(m: Maze) -> bool:
    q = [(0, 0)]
    seen = set(q)
    while q:
        x, y = q.pop()
        for d, nx, ny in neigh(x, y):
            if (nx, ny) in seen or wall(m, x, y, d):
                continue
            seen.add((nx, ny))
            q.append((nx, ny))
    return len(seen) == config.size * config.size

def violates(m: Maze) -> bool:
    # 直線の長さチェック
    for y in range(config.size):
        run = 1
        for x in range(config.size-1):
            run = run + 1 if not m.v[y][x] else 1
            if run > config.max_straight:
                return True
    for x in range(config.size):
        run = 1
        for y in range(config.size-1):
            run = run + 1 if not m.h[y][x] else 1
            if run > config.max_straight:
                return True
    # 2×2 通路
    for y in range(config.size-1):
        for x in range(config.size-1):
            if (
                not m.v[y][x] and not m.v[y+1][x] and
                not m.h[y][x] and not m.h[y][x+1]
            ):
                return True
    # 分岐距離
    deg = [[0]*config.size for _ in range(config.size)]
    for y in range(config.size):
        for x in range(config.size-1):
            if not m.v[y][x]:
                deg[y][x] += 1
                deg[y][x+1] += 1
    for y in range(config.size-1):
        for x in range(config.size):
            if not m.h[y][x]:
                deg[y][x] += 1
                deg[y+1][x] += 1
    visited: set[Cell] = set()
    for y in range(config.size):
        for x in range(config.size):
            if (x, y) in visited:
                continue
            if deg[y][x] == 2:
                chain = [(x, y)]
                visited.add((x, y))
                ends = [(x, y)]
                while ends:
                    cx, cy = ends.pop()
                    for d, nx, ny in neigh(cx, cy):
                        if (nx, ny) in visited or wall(m, cx, cy, d):
                            continue
                        if deg[ny][nx] == 2:
                            chain.append((nx, ny))
                            visited.add((nx, ny))
                            ends.append((nx, ny))
                if len(chain) > config.max_branch_dist:
                    return True
            else:
                visited.add((x, y))
    return False

# -------------------- グラフと径計算 --------------------
def maze_to_graph(m: Maze) -> Adj:
    g: Adj = {(x, y): [] for y in range(config.size) for x in range(config.size)}
    for y in range(config.size):
        for x in range(config.size):
            for d, nx, ny in neigh(x, y):
                if not wall(m, x, y, d):
                    g[(x, y)].append((nx, ny))
    return g

def bfs_dist(g: Adj, src: Cell) -> Dict[Cell, int]:
    dist = {src: 0}
    dq = deque([src])
    while dq:
        v = dq.popleft()
        for u in g[v]:
            if u not in dist:
                dist[u] = dist[v] + 1
                dq.append(u)
    return dist

def graph_diameter(g: Adj) -> Tuple[int, Cell, Cell]:
    v0 = next(iter(g))
    d0 = bfs_dist(g, v0)
    A = max(d0, key=d0.get)
    d1 = bfs_dist(g, A)
    B = max(d1, key=d1.get)
    return d1[B], A, B

# -------------------- 径制約付き生成 --------------------
def generate_maze_with_diameter() -> Tuple[Maze, Tuple[int, Cell, Cell]]:
    attempts = 0
    while True:
        attempts += 1
        if attempts > config.max_attempts_per_maze:
            print(
                f"⚠️  {attempts} attempts > limit."
                f" Consider raising config.max_diameter ({config.max_diameter}).",
                file=sys.stderr,
            )
            attempts = 0
        m = generate_tree()
        add_loops(m)
        if not (is_connected(m) and not violates(m)):
            continue
        dia_len, A, B = graph_diameter(maze_to_graph(m))
        if dia_len <= config.max_diameter:
            return m, (dia_len, A, B)

# -------------------- JSON 変換 --------------------
def maze_to_json(m: Maze, idx: int, dia: Tuple[int, Cell, Cell]) -> Dict:
    dia_len, A, B = dia
    v_walls = [[x, y] for y in range(config.size) for x in range(config.size-1) if m.v[y][x]]
    h_walls = [[x, y] for y in range(config.size-1) for x in range(config.size) if m.h[y][x]]
    return {
        "id": f"maze{idx:03}",
        "size": config.size,
        "max_straight": config.max_straight,
        "max_branch_dist": config.max_branch_dist,
        "loop_rate": config.loop_rate,
        "max_diameter": config.max_diameter,
        "diameter": dia_len,
        "dia_endpoints": [A, B],
        "v_walls": v_walls,
        "h_walls": h_walls,
    }

# -------------------- バッチ生成 --------------------
def _generate_single(cfg: Config, idx: int) -> Tuple[Dict, str]:
    """1 つの迷路を生成して JSON と文字描画を返す"""
    m, dia_info = generate_maze_with_diameter()
    ascii_maze = render_maze(m)
    return maze_to_json(m, idx, dia_info), ascii_maze


def generate_batch(cfg: Config, workers: int) -> List[Dict]:
    """複数迷路を並列で生成する"""
    if cfg.seed is not None:
        random.seed(cfg.seed)

    out: List[Dict] = [None] * cfg.maze_count
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_generate_single, cfg, i): i for i in range(1, cfg.maze_count + 1)}
        for fut in as_completed(futures):
            idx = futures[fut]
            data, ascii_maze = fut.result()
            print(f"\n[{idx}/{cfg.maze_count}] generated (diam = {data['diameter']})")
            print(ascii_maze)
            out[idx - 1] = data

    return out

# 出力を保存

def save_batch(mazes: List[Dict]):
    """生成結果を JSON ファイルに保存"""
    config.out_dir.mkdir(exist_ok=True)
    path = config.out_dir / f"maze_{config.size}.json"

    # Path.write_text で JSON 文字列を書き込む
    path.write_text(
        json.dumps(mazes, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"✅ Saved {len(mazes)} maze(s) → {path}")

# -------------------- main --------------------
def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解釈"""
    parser = argparse.ArgumentParser(description="迷路を生成して JSON 出力")
    parser.add_argument("-n", "--count", type=int, default=config.maze_count,
                        help="生成する迷路の個数")
    parser.add_argument("-w", "--workers", type=int,
                        default=os.cpu_count() or 1,
                        help="並列実行するプロセス数")
    parser.add_argument("--seed", type=int, default=None,
                        help="乱数シード。指定すると再現可能")
    return parser.parse_args()


def main():
    args = parse_args()
    config.maze_count = args.count
    config.seed = args.seed
    batch = generate_batch(config, args.workers)
    save_batch(batch)

if __name__ == "__main__":
    main()
