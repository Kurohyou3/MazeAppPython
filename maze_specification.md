
# 暗闇迷路アプリ – 迷路データ仕様まとめ
生成日: 2025-06-24 20:24:45

---

## 1. 基本パラメータ
| 定数 | 説明 | 既定値 |
|------|------|--------|
| `WIDTH`, `HEIGHT` | 盤面のセル数 (正方形) | `11` |
| `START` | スタート座標 (中央) | `(WIDTH//2, HEIGHT//2)` → `(5,5)` |
| `TURN_LIMIT` | 最短経路上の曲がり角上限 | `30` |
| `MIN_PATH_LEN` | スタート→ゴール最短距離の下限 (セル数) | `16` |

> `TURN_LIMIT`・`MIN_PATH_LEN` は変数で管理しており、値を変更するだけで制約が動的に変わります。

---

## 2. 制約一覧
1. 直交壁禁止（2 連横壁 × 2 連縦壁 の交差を除去）  
2. 壁の直線長 ≤ 4 （縦方向・横方向とも）  
3. 通路の直進長 ≤ 4  
4. 2 × 2 の完全開放通路禁止  
5. 最短経路長 ≥ `MIN_PATH_LEN`  
6. 曲がり角数 ≤ `TURN_LIMIT`

---

## 3. アルゴリズム概要
1. **DFS**で木構造迷路を生成  
2. 後処理  
   - `remove_wall_crossings` … 直交壁排除  
   - `shorten_wall_runs` … 壁直線長調整  
3. 制約検査に合格するまで無制限再生成  

---

## 4. JSON 仕様

```json
{
  "id": "maze001",
  "size": 11,
  "start": [5, 5],
  "goal": [x, y],
  "v_walls": [[x1, y1], ...],
  "h_walls": [[x2, y2], ...]
}
```

外周壁はクライアント側が暗黙追加し、配列は 0‑index。

---

## 5. ファイル出力規則
```python
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"maze_{WIDTH}x{HEIGHT}_T{TURN_LIMIT}_L{MIN_PATH_LEN}_{timestamp}.json"
```
例: `maze_11x11_T30_L16_20250704_143213.json`

---

## 6. 主要関数
| 関数 | 用途 |
|------|------|
| `generate_base_maze()` | DFS 生成 |
| `remove_wall_crossings()` | 直交壁排除 |
| `shorten_wall_runs()` | 壁長調整 |
| `has_open_square()` | 2×2 通路チェック |
| `corridor_run_ok()` | 通路直線長チェック |
| `shortest_path()` | 最短経路 (BFS) |
| `count_turns()` | 曲がり角数 |
| `generate_valid_maze()` | 1 迷路生成・JSON 化 |
| `maze_to_json()` | 壁配列→JSON |

---

## 7. 量産スクリプト雛形
```python
MAZE_COUNT = 10
mazes = [generate_valid_maze(f"maze{str(i).zfill(3)}") for i in range(1, MAZE_COUNT+1)]

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"maze_{WIDTH}x{HEIGHT}_T{TURN_LIMIT}_L{MIN_PATH_LEN}_{timestamp}.json"
os.makedirs("mazefiles", exist_ok=True)
with open(os.path.join("mazefiles", file_name), "w", encoding="utf-8") as f:
    json.dump(mazes, f, ensure_ascii=False, indent=2)
```

---

## 8. 拡張アイデア
- 難易度パラメータ (行き止まり数上限など) の追加  
- シード固定 (`random.seed`) による再現性  
- 生成済み迷路のメタデータ付加 (生成日時・パラメータ)  
