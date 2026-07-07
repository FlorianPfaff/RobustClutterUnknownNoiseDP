"""Run the default structured-clutter comparison experiment.

This is a thin convenience wrapper around the package CLI. For custom sweeps,
use the installed console command directly:

```bash
robust-clutter-dp-experiment --help
```
"""

from __future__ import annotations

from robust_clutter_dp.cli import main as cli_main


def main() -> int:
    return cli_main(
        [
            "--scenarios",
            "hotspot,no_hotspot_control,near_hotspot_crossing",
            "--methods",
            "uniform,grid,dp,oracle",
            "--seeds",
            "0:5",
            "--output",
            "all",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
