"""
暗闇迷路 JSON 生成スクリプト
---------------------------------------
* 盤面サイズ      : WIDTH × HEIGHT（正方形）
* スタート        : 盤面中央
* ゴール          : 外周からランダム
* 制約一覧
    1. 直交壁（十字壁）禁止
    2. 壁の直線長 ≤ 4
    3. 通路の直進長 ≤ 4
    4. 2×2 完全開放通路禁止
    5. 最短経路長 ≥ MIN_PATH_LEN
    6. 曲がり角数 ≤ TURN_LIMIT
* 出力ファイル例 : maze_11x11_T30_L16_20250704_143213.json
"""

# -------------------- 基本定数 --------------------
import os, json, random, datetime
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict

WIDTH, HEIGHT   = 5 ,5
START           = (WIDTH // 2, HEIGHT // 2)   # (5,5)
TURN_LIMIT      = 30
MIN_PATH_LEN    = 7
MAZE_COUNT      = 10                          # 生成個数
OUT_DIR         = "mazefiles"                 # 出力ディレクトリ

# -------------------- データ構造 --------------------
@dataclass
class Maze:
    vert_walls:  List[List[bool]]  # 垂直: (y,x) True=壁あり between (x,y)-(x+1,y)
    horiz_walls: List[List[bool]]  # 水平: (y,x) True=壁あり between (x,y)-(x,y+1)

# -------------------- ユーティリティ --------------------
def cell_neighbors(x: int, y: int) -> List[Tuple[str, int, int]]:
    nb = []
    if x > 0:          nb.append(('L', x-1, y))
    if x < WIDTH-1:    nb.append(('R', x+1, y))
    if y > 0:          nb.append(('U', x, y-1))
    if y < HEIGHT-1:   nb.append(('D', x, y+1))
    return nb

# ---------- ❶ DFS で木構造迷路を生成 ----------
def generate_base_maze() -> Maze:
    visited = [[False]*WIDTH for _ in range(HEIGHT)]
    vert    = [[True]*(WIDTH-1) for _ in range(HEIGHT)]
    horiz   = [[True]*WIDTH      for _ in range(HEIGHT-1)]

    stack = [START]
    visited[START[1]][START[0]] = True

    while stack:
        x, y = stack[-1]
        unvisited = [(d, nx, ny) for d, nx, ny in cell_neighbors(x, y)
                     if not visited[ny][nx]]
        if unvisited:
            d, nx, ny = random.choice(unvisited)
            if d == 'L': vert[y][x-1]  = False
            if d == 'R': vert[y][x]    = False
            if d == 'U': horiz[y-1][x] = False
            if d == 'D': horiz[y][x]   = False
            visited[ny][nx] = True
            stack.append((nx, ny))
        else:
            stack.pop()
    return Maze(vert, horiz)

# ---------- ❷ 後処理: 直交壁排除 ----------
def remove_wall_crossings(m: Maze) -> None:
    changed = True
    while changed:
        changed = False
        for y in range(HEIGHT-1):
            for x in range(WIDTH-1):
                if (m.vert_walls[y][x] and m.vert_walls[y+1][x] and
                    m.horiz_walls[y][x] and m.horiz_walls[y][x+1]):
                    # 4 辺のうちランダムに 1 辺を除去
                    if random.random() < 0.5:
                        m.vert_walls[y][x] = False
                    else:
                        m.horiz_walls[y][x] = False
                    changed = True

# ---------- ❸ 後処理: 壁直線長を 4 で分断 ----------
def shorten_wall_runs(m: Maze, max_len: int = 4) -> None:
    # 水平壁
    for y in range(HEIGHT-1):
        run = 0
        for x in range(WIDTH):
            if m.horiz_walls[y][x]:
                run += 1
                if run > max_len:
                    m.horiz_walls[y][x] = False
                    run = 0
            else:
                run = 0
    # 垂直壁
    for x in range(WIDTH-1):
        run = 0
        for y in range(HEIGHT):
            if m.vert_walls[y][x]:
                run += 1
                if run > max_len:
                    m.vert_walls[y][x] = False
                    run = 0
            else:
                run = 0

# ---------- ❹ 制約チェック ----------
def has_open_square(m: Maze) -> bool:
    for y in range(HEIGHT-1):
        for x in range(WIDTH-1):
            if (not m.vert_walls[y][x]   and not m.vert_walls[y+1][x] and
                not m.horiz_walls[y][x] and not m.horiz_walls[y][x+1]):
                return True
    return False

def corridor_run_ok(m: Maze, max_len: int = 4) -> bool:
    # 横方向
    for y in range(HEIGHT):
        run = 1
        for x in range(WIDTH-1):
            run = run + 1 if not m.vert_walls[y][x] else 1
            if run > max_len: return False
    # 縦方向
    for x in range(WIDTH):
        run = 1
        for y in range(HEIGHT-1):
            run = run + 1 if not m.horiz_walls[y][x] else 1
            if run > max_len: return False
    return True

# ---------- ❺ 最短経路 & 曲がり角 ----------
def shortest_path(m: Maze, start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    prev: Dict[Tuple[int,int], Tuple[int,int]|None] = {start: None}
    dq = deque([start])
    while dq:
        x, y = dq.popleft()
        if (x, y) == goal:
            break
        for d, nx, ny in cell_neighbors(x, y):
            if   d=='L' and m.vert_walls[y][x-1]:  continue
            elif d=='R' and m.vert_walls[y][x]:    continue
            elif d=='U' and m.horiz_walls[y-1][x]: continue
            elif d=='D' and m.horiz_walls[y][x]:   continue
            if (nx, ny) in prev: continue
            prev[(nx, ny)] = (x, y)
            dq.append((nx, ny))
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    return list(reversed(path))

def count_turns(path: List[Tuple[int,int]]) -> int:
    if len(path) < 3:
        return 0
    turns = 0
    dx_prev, dy_prev = path[1][0]-path[0][0], path[1][1]-path[0][1]
    for (px,py), (qx,qy) in zip(path[1:-1], path[2:]):
        dx, dy = qx-px, qy-py
        if (dx,dy) != (dx_prev,dy_prev):
            turns += 1
        dx_prev, dy_prev = dx, dy
    return turns

# ---------- ❻ ゴール位置 ----------
def random_goal() -> Tuple[int,int]:
    boundary = (
        [(x, 0) for x in range(WIDTH)] +
        [(x, HEIGHT-1) for x in range(WIDTH)] +
        [(0, y) for y in range(1, HEIGHT-1)] +
        [(WIDTH-1, y) for y in range(1, HEIGHT-1)]
    )
    if START in boundary:
        boundary.remove(START)
    return random.choice(boundary)

# ---------- ❼ 迷路 → JSON ----------
def maze_to_json(m: Maze, maze_id: str, goal: Tuple[int,int]) -> Dict:
    v_list = [[x, y] for y in range(HEIGHT) for x in range(WIDTH-1) if m.vert_walls[y][x]]
    h_list = [[x, y] for y in range(HEIGHT-1) for x in range(WIDTH) if m.horiz_walls[y][x]]
    return {
        "id": maze_id,
        "size": WIDTH,
        "start": [START[0], START[1]],
        "goal": [goal[0], goal[1]],
        "v_walls": v_list,
        "h_walls": h_list
    }

# ---------- ❽ 単一迷路を制約クリアまで生成 ----------
def generate_valid_maze(maze_id: str) -> Dict:
    while True:
        goal = random_goal()
        m = generate_base_maze()
        remove_wall_crossings(m)
        shorten_wall_runs(m, 4)

        if has_open_square(m):               continue
        if not corridor_run_ok(m, 4):        continue

        path = shortest_path(m, START, goal)
        if len(path)-1 < MIN_PATH_LEN:       continue
        if count_turns(path) > TURN_LIMIT:   continue

        return maze_to_json(m, maze_id, goal)

# -------------------- メイン処理 --------------------
def main():
    random.seed()  # 必要ならシード固定
    mazes = [generate_valid_maze(f"maze{str(i).zfill(3)}")
             for i in range(1, MAZE_COUNT+1)]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"maze_{WIDTH}x{HEIGHT}_T{TURN_LIMIT}_L{MIN_PATH_LEN}_{timestamp}.json"
    os.makedirs(OUT_DIR, exist_ok=True)
    file_path = os.path.join(OUT_DIR, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(mazes, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(mazes)} mazes saved to: {file_path}")

# -------------------- 実行 --------------------
if __name__ == "__main__":
    main()
