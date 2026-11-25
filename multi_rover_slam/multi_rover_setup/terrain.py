"""Terrain presets and a height field sampled from the spawned terrain mesh.

:func:`build_scenery` imports SRB lazily (after Kit starts);
:class:`TerrainHeightField` is pure numpy apart from :meth:`from_stage`.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from .config import TerrainCfg

logger = logging.getLogger("multi_rover_setup")

# kind → (domain, real-DEM site or None)
TERRAIN_PRESETS: dict[str, tuple[str, str | None]] = {
    "moon_surface": ("moon", None),
    "mars_surface": ("mars", None),
    "apollo17": ("moon", "apollo17"),
    "apollo15": ("moon", "apollo15"),
    "jezero": ("mars", "jezero"),
    "jezero_v2": ("mars", "jezero_v2"),
    "flat": ("moon", None),
}


def terrain_domain(cfg: TerrainCfg) -> str:
    return TERRAIN_PRESETS[cfg.kind][0]


def terrain_half_extent(cfg: TerrainCfg) -> float:
    """Half-size [m] of the usable terrain area around the origin."""
    site = TERRAIN_PRESETS[cfg.kind][1]
    if cfg.kind == "flat":
        return 1e3
    if site is None:
        return 0.5 * cfg.size_m
    return 0.5 * (cfg.patch_size_m or 98.0)


def procedural_defaults(size_m: float) -> tuple[float, float]:
    """``(density, flat_area_size)`` with the same scaling SRB uses for its variants."""
    dyn = max(1, min(10, 2 * round(min(8.0, math.pow(size_m, 0.4)) / 2)))
    return 0.02 * dyn**2, 0.8 * dyn**1.2


def texture_resolution(px: int) -> dict:
    """SimForge bake resolutions keeping SRB's ratios (albedo/normal = ``px``)."""
    from simforge import BakeType

    px = int(px)
    return {
        BakeType.ALBEDO: px,
        BakeType.NORMAL: px,
        BakeType.ROUGHNESS: max(px // 2, 64),
        BakeType.METALLIC: max(px // 4, 64),
        BakeType.EMISSION: max(px // 8, 64),
    }


def build_scenery(cfg: TerrainCfg):
    """Return an SRB ``Scenery`` instance for ``cfg``."""
    domain, site = TERRAIN_PRESETS[cfg.kind]
    if cfg.kind == "flat":
        from srb.assets.scenery.terrain import GroundPlane

        return GroundPlane()
    if site is None:
        from srb.assets.scenery.terrain import MarsSurface, MoonSurface

        density, flat_area = procedural_defaults(cfg.size_m)
        surface_cls = MoonSurface if domain == "moon" else MarsSurface
        size = float(cfg.size_m)
        return surface_cls(
            scale=(size, size, 0.1 * size),
            texture_resolution=texture_resolution(cfg.texture_px),
            density=cfg.density if cfg.density is not None else density,
            flat_area_size=cfg.flat_area_m if cfg.flat_area_m is not None else flat_area,
        )

    from srb.assets.scenery.real_terrain import RealTerrain

    kwargs = {"body": domain, "site": site, "lod": cfg.lod, "augment": cfg.augment}
    if cfg.patch_size_m is not None:
        kwargs["patch_size_m"] = float(cfg.patch_size_m)
    try:
        return RealTerrain(**kwargs)
    except Exception as err:
        raise RuntimeError(
            f"Real terrain {site!r} (lod={cfg.lod}, patch={cfg.patch_size_m}) is not available: {err}\n"
            f"Check `srb asset info terrain {site}`; a new lod/patch needs `srb asset download terrain {site}`"
        ) from err


class TerrainHeightField:
    """Max-z grid over terrain vertices for spawn-height queries."""

    def __init__(self, points_w: np.ndarray, cell_m: float = 0.25):
        points = np.asarray(points_w, dtype=float).reshape(-1, 3)
        if len(points) == 0:
            raise ValueError("TerrainHeightField needs at least one point")
        self.cell = float(cell_m)
        self.origin = points[:, :2].min(axis=0)
        ij = np.floor((points[:, :2] - self.origin) / self.cell).astype(np.int64)
        self.shape = tuple(ij.max(axis=0) + 1)
        grid = np.full(self.shape, -np.inf)
        np.maximum.at(grid, (ij[:, 0], ij[:, 1]), points[:, 2])
        self.grid = grid
        self.num_points = len(points)

    def height_at(self, x: float, y: float, radius_m: float = 0.0, max_search_m: float = 10.0) -> float | None:
        """Highest terrain point within ``radius_m`` of ``(x, y)``; ``None`` off the mesh.

        On coarse meshes (vertices further apart than the footprint) the window
        grows until it contains vertices, up to ``max_search_m``.
        """
        ci, cj = np.floor((np.array((x, y)) - self.origin) / self.cell).astype(np.int64)
        if not (0 <= ci < self.shape[0] and 0 <= cj < self.shape[1]):
            return None
        reach = max(int(math.ceil(radius_m / self.cell)), 1)
        limit = max(int(math.ceil(max_search_m / self.cell)), reach)
        while True:
            i0, i1 = max(ci - reach, 0), min(ci + reach + 1, self.shape[0])
            j0, j1 = max(cj - reach, 0), min(cj + reach + 1, self.shape[1])
            window = self.grid[i0:i1, j0:j1]
            finite = window[np.isfinite(window)]
            if finite.size:
                return float(finite.max())
            if reach >= limit:
                return None
            reach = min(2 * reach, limit)

    @classmethod
    def from_stage(cls, stage, root_path: str, cell_m: float = 0.25) -> TerrainHeightField | None:
        """Height field over the world-space vertices of every mesh under ``root_path``."""
        meshes = mesh_points_from_stage(stage, root_path)
        if not meshes:
            return None
        return cls(np.concatenate([points for _, points in meshes], axis=0), cell_m=cell_m)


def mesh_points_from_stage(stage, root_path: str) -> list[tuple[str, np.ndarray]]:
    """``(prim path, (N, 3) world-space vertices)`` of every mesh under ``root_path``; empty if none."""
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return []
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    meshes = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points:
            continue
        local = np.asarray(points, dtype=float)
        mat = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=float)
        # USD matrices are row-major with row vectors: p_w = [p, 1] @ M
        meshes.append((str(prim.GetPath()), local @ mat[:3, :3] + mat[3, :3]))
    return meshes


###########################
## Saved terrain geometry ##
###########################

TERRAIN_FILE = "terrain.npz"
"""Terrain vertices in the run folder, for drawing (e.g. ``overseer_viz``); written by ``place_on_terrain``."""


def save_terrain_geometry(path, meshes: list[tuple[str, np.ndarray]]) -> None:
    """Write world-space mesh vertices as ``points`` (N, 3), ``mesh_id`` (N,) and ``mesh_paths`` (K,)."""
    if not meshes:
        raise ValueError("save_terrain_geometry needs at least one mesh")
    points = np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1, 3) for _, p in meshes], axis=0)
    mesh_id = np.concatenate([np.full(len(p), i, dtype=np.int32) for i, (_, p) in enumerate(meshes)])
    np.savez_compressed(path, points=points, mesh_id=mesh_id, mesh_paths=np.array([name for name, _ in meshes]))


def load_terrain_geometry(path) -> list[tuple[str, np.ndarray]]:
    """Inverse of :func:`save_terrain_geometry`: ``(prim path, (N, 3) float64 vertices)`` per mesh."""
    with np.load(path) as data:
        points, mesh_id, names = data["points"].astype(float), data["mesh_id"], data["mesh_paths"]
    return [(str(name), points[mesh_id == i]) for i, name in enumerate(names)]
