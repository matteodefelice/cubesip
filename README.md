# cubesip

Retrieve remote sensing and climate data for a site or region, addressed by a
[geohash](https://en.wikipedia.org/wiki/Geohash), from openEO, Earth Data Hub or local
Zarr/NetCDF/GeoTIFF, with a shared CSV manifest that avoids downloading the same data twice.

> Status: scaffold. See [plan.MD](plan.MD) for the goals and
> [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the design and milestones.

## Setup

```bash
uv sync
cp cubesip.example.yaml cubesip.yaml   # then add your credentials (git-ignored)
uv run pytest
```

## Usage (target API)

```python
import cubesip

req = cubesip.Request.from_json("examples/request_s2_ndvi.json")
report = cubesip.check(req)  # dataset found? already cached? estimated size?
result = cubesip.run(req)  # result.path (NetCDF) + result.dataset (lazy xarray)
```

Request documents follow [schemas/request.schema.json](schemas/request.schema.json).
