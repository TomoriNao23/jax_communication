# HaloExchange 优化说明

## 优化 B：`lax.cond` 替换 `jnp.where` 全数组广播

### 问题

原版在 `_one_round` 中，每条边都对**所有** tile 计算两个完整的 `u` 副本：

```python
# 原版
u_if_A = _unpack_and_write(u, recv_data, dA, tid)  # 所有 tile 均计算完整副本
u_if_B = _unpack_and_write(u, recv_data, dB, tid)
u = jnp.where(tile_id == tA, u_if_A, u)            # 全数组广播选择
u = jnp.where(tile_id == tB, u_if_B, u)
```

`jnp.where(cond, a, b)` 要求 `a` 和 `b` 在调用前完整求值。对于 `.at[].set()` 这类写操作，XLA 的 DCE 通常不生效，导致每条边都产生 `(nx+2h)×(ny+2h)` 大小的冗余数组运算。

打包侧同理：

```python
send_data = jnp.where(tile_id == tA, _pack_edge(u, dA), send_data)
# 所有 tile 均计算 _pack_edge(u, dA)，jnp.where 仅选择结果
```

### 方案

用 `lax.cond` 替换 `jnp.where`：

```python
# 优化 B：写入侧
lax.cond(
    tile_id == target_tile,
    lambda _: u.at[h:-h, -h:].set(arr),  # 只在目标 tile 执行
    lambda _: u,                           # 其余 tile 直接返回原 u
    None,
)

# 优化 B：打包侧
send_data = lax.cond(
    tile_id == tA,
    lambda _: cls._pack_edge(u, dA),  # 只在 tA 执行
    lambda _: send_data,
    None,
)
```

`lax.cond` 保证只执行命中分支的计算；两分支均返回 `u` 形状，无形状约束问题。

### 残留问题

优化 B 中打包侧仍对 `edges_in_round` 中**所有边**循环：

```python
for (tA, dA, tB, dB, _) in edges_in_round:
    packed_A = cls._pack_edge(u, dA)           # Python 级求值，节点仍进入 trace
    lax.cond(tile_id == tA, lambda _, p=packed_A: p, ...)
    packed_B = cls._pack_edge(u, dB)           # 同上
    lax.cond(tile_id == tB, lambda _, p=packed_B: p, ...)
```

`packed_A` / `packed_B` 在 `lax.cond` 外部提前求值，XLA tracer 仍会为每个 tile 建立所有 `_pack_edge` 的计算节点。`lax.cond` 的 DCE 在 `vmap` 下行为取决于 XLA 版本，不保证消除。

---

## 优化 C：路由表消除冗余打包（在 B 基础上叠加）

### 问题根源

优化 B 的 `_one_round` 以「边」为单位循环，每条边展开 2 个 `_pack_edge` 节点进入 trace 图。一轮有 `k` 条边，则产生 `2k` 个节点，而本 tile 实际只需要 1 个。

图着色保证每 tile 每轮**至多参与 1 条边**，这一静态事实在优化 B 中未被利用。

### 关键性质：`send_dir == write_halo_dir`

对边 `(tA, dA, tB, dB, tid)`，ppermute 后：

- tile_A 收到 tile_B 打包的 `dB` 数据，需写入 tile_A 的 `dA` halo
- tile_A 自身发送的是 `dA` 方向

因此 `send_dir = dA`，`write_halo_dir = dA`，**两者恒相等**。路由表只需一个 `direction` 字段。

### 路由表结构

```
_schedule[round_idx][tile_id] = (direction: int, partner: int, tid: int)
                               | None   # IDLE：本轮不参与通信
```

`configure()` 时由 `_build_schedule()` 一次性预计算，运行时直接查表。

### 方案

`_one_round` 改为按路由表以「tile」为单位展开，每个 tile 只生成 1 个 `_pack_edge` 节点（IDLE tile 生成 0 个）：

```python
# 优化 C：打包侧
send_data = jnp.zeros((h, n_pad), dtype=u.dtype)
for t in range(cls.ntile):
    entry = round_sched[t]
    if entry is None:
        # IDLE：发送零缓冲（ppermute 要求双射，仍需参与置换）
        send_data = lax.cond(
            tile_id == t,
            lambda _: jnp.zeros((h, n_pad), dtype=u.dtype),
            lambda _, s=send_data: s,
            None,
        )
    else:
        direction, _, _ = entry
        # _pack_edge 在 lambda 内部，只有本 tile 命中时才进入 trace
        send_data = lax.cond(
            tile_id == t,
            lambda _, d=direction: cls._pack_edge(u, d),
            lambda _, s=send_data: s,
            None,
        )

# 优化 C：写入侧
for t in range(cls.ntile):
    entry = round_sched[t]
    if entry is None:
        continue  # IDLE：完全不生成写入节点
    direction, _, tid = entry
    u = cls._write_halo_if_tile(u, recv_data, direction, tid, t, tile_id)
```

`_pack_edge` 放在 `lambda` **内部**是关键：只有 `tile_id == t` 时该分支才被 trace，其余 tile 的 false_fn 中完全不存在 `_pack_edge` 节点。

### trace 节点数对比

以 Round 0（2 条边，4 个 active tile，2 个 IDLE tile）为例：

|          | 打包侧 `_pack_edge` 节点 | 写入侧节点 |
|----------|--------------------------|------------|
| 原版     | `2 × 2 = 4`，jnp.where 不保证消除 | `2 × 2 = 4` 完整 u 副本 |
| 优化 B   | 4 个进入 trace，lax.cond 可能消除 2 个 | 每条边 2 次 lax.cond，共 4 次 |
| 优化 C   | **1**（本 tile 自己的） | **1**（本 tile 自己的），IDLE 为 0 |

全局（6 轮）：优化 C 相比优化 B 减少约 **12 个**冗余 `_pack_edge` trace 节点。

### 收益场景

优化 C 的收益在以下场景更显著：

- 每轮边数较多（`edges_in_round` 较大）
- ntile 较大
- JIT 编译频率高（编译产物更小 → 重新编译更快）
- `_pack_edge` 计算量大（halo 宽、网格精细时）

---

## 两次优化对比总览

| 维度 | 原版 | 优化 B | 优化 C |
|------|------|--------|--------|
| 写入侧选择 | `jnp.where` 全数组 | `lax.cond` | `lax.cond`（同 B）|
| 打包节点数/轮 | `2k`，不保证消除 | `2k` 进 trace，部分消除 | **1**（本 tile）|
| IDLE tile 写入 | `jnp.where` 选择 | `lax.cond` false_fn | **完全跳过**（continue）|
| 编译产物大小 | 最大 | 中 | 最小 |
| 实现额外开销 | — | 无 | `configure()` 多一次静态遍历（一次性）|
