# 迷路生成 & 表示サンプル（Jupyter Notebook セル 1 つで完結）

import random
import json
from collections import deque

# ─────────────────────────────────────────
# ① パラメータ
# ─────────────────────────────────────────
SIZE = 10  # 10×10
START = (1, 1)
GOAL = (8, 8)
MAX_STRAIGHT = 3  # 連続壁の上限
WALL_PROB = 0.35  # 壁候補にする確率
MAX_ATTEMPTS = 10_000  # 壁配置試行回数


# ─────────────────────────────────────────
# ② ユーティリティ
# ─────────────────────────────────────────
def reachable(grid, s, g):
    """BFS で s→g が到達可能か判定（壁＝1, 通路＝0）"""
    q = deque([s])
    seen = {s}
    while q:
        r, c = q.popleft()
        if (r, c) == g:
            return True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < SIZE and 0 <= nc < SIZE and grid[nr][nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return False


def straight_ok(grid, r, c):
    """壁を置いても直線 4 連続にならないか確認"""
    # 行方向
    cnt = 1
    cc = c - 1
    while cc >= 0 and grid[r][cc] == 1:
        cnt += 1;
        cc -= 1
    cc = c + 1
    while cc < SIZE and grid[r][cc] == 1:
        cnt += 1;
        cc += 1
    if cnt > MAX_STRAIGHT:
        return False

    # 列方向
    cnt = 1
    rr = r - 1
    while rr >= 0 and grid[rr][c] == 1:
        cnt += 1;
        rr -= 1
    rr = r + 1
    while rr < SIZE and grid[rr][c] == 1:
        cnt += 1;
        rr += 1
    return cnt <= MAX_STRAIGHT


def print_maze(grid):
    """S, G, ■, ' ' でコンソール出力"""
    for r in range(SIZE):
        row_str = ''
        for c in range(SIZE):
            if (r, c) == START:
                row_str += 'S'
            elif (r, c) == GOAL:
                row_str += 'G'
            elif grid[r][c] == 1:
                row_str += '■'
            else:
                row_str += ' '
        print(row_str)


# ─────────────────────────────────────────
# ③ 迷路生成
# ─────────────────────────────────────────
def generate_maze():
    # 0 = 通路, 1 = 壁
    grid = [[0] * SIZE for _ in range(SIZE)]
    visited = {START}
    path = [START]
    cur = START

    # --- (A) スタート⇒ゴールをつなぐ一次経路 ---
    while cur != GOAL:
        r, c = cur
        nbrs = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= r + dr < SIZE and 0 <= c + dc < SIZE and (r + dr, c + dc) not in visited]
        if not nbrs:  # 行き止まり → バックトラック
            for prev in reversed(path):
                r, c = prev
                nbrs = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                        if 0 <= r + dr < SIZE and 0 <= c + dc < SIZE and (r + dr, c + dc) not in visited]
                if nbrs:
                    cur = prev
                    break
        else:
            nxt = random.choice(nbrs)
            visited.add(nxt)
            path.append(nxt)
            cur = nxt

    # --- (B) 壁をランダムに追加 ---
    attempts = 0
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        r = random.randint(0, SIZE - 1)
        c = random.randint(0, SIZE - 1)

        # 置けない条件（スタート／ゴール／既に壁／経路マス）
        if (r, c) in (START, GOAL) or grid[r][c] == 1 or (r, c) in path:
            continue
        if random.random() > WALL_PROB:
            continue
        if not straight_ok(grid, r, c):
            continue

        # 仮に壁を置いて到達可能性を再評価
        grid[r][c] = 1
        if not reachable(grid, START, GOAL):
            grid[r][c] = 0  # 行き止まりになるなら撤回

    return grid


# ─────────────────────────────────────────
# ④ 実行 & 出力
# ─────────────────────────────────────────
maze = generate_maze()
print_maze(maze)

