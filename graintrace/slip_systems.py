# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Slip-system definitions for the NEML2 crystal-plasticity decks.

The CPFE deck (``cpfe_base/neml2_cpfe.i``) and the Taylor/calibration deck
(``cpfe_base/neml2_cpfe_calibration.i``) describe slip with NEML2's
``CubicCrystal``: one or more slip *families*, each a Miller direction and a
Miller plane normal. NEML2 expands every family over the cubic point group, so
``(<110>, {111})`` gives the 12 FCC systems. The two deck lines that hold the
families carry ``# @SLIP directions`` / ``# @SLIP planes`` markers;
:meth:`SlipSystems.apply_to_deck` rewrites them.

Pass a preset name (:data:`SLIP_SYSTEM_PRESETS`) or the families directly::

    SlipSystems.resolve("bcc")                              # {110}<111>, 12 systems
    SlipSystems.resolve([((1, 1, 1), (1, 1, 0)),
                         ((1, 1, 1), (1, 1, 2))])           # 24 systems
    SlipSystems.resolve([{"direction": [1, 1, 1], "plane": [1, 2, 3]}])

Only the cubic crystal class is supported: NEML2 v3 exposes ``CubicCrystal`` as
its only crystal-geometry input block (HCP and other classes need upstream
support).
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Union

Miller = Tuple[int, int, int]
Family = Tuple[Miller, Miller]  # (slip direction, slip-plane normal)

#: Named slip-system sets: name -> tuple of (direction, plane) families.
SLIP_SYSTEM_PRESETS = {
    "fcc": (((1, 1, 0), (1, 1, 1)),),  # <110>{111}, 12 systems
    "bcc": (((1, 1, 1), (1, 1, 0)),),  # <111>{110}, 12 systems
    "bcc_112": (((1, 1, 1), (1, 1, 2)),),  # <111>{112}, 12 systems
    "bcc_110_112": (((1, 1, 1), (1, 1, 0)), ((1, 1, 1), (1, 1, 2))),  # 24 systems
    "bcc_110_112_123": (
        ((1, 1, 1), (1, 1, 0)),
        ((1, 1, 1), (1, 1, 2)),
        ((1, 1, 1), (1, 2, 3)),
    ),  # 48 systems
}

#: Preset used when nothing is specified (the shipped decks' historical slip).
DEFAULT_SLIP_SYSTEMS = "fcc"

_MARKER_RE = {
    "directions": re.compile(
        r"(expr\s*=\s*')MillerIndex\(torch\.tensor\([^']*\)\)(')(\s*#\s*@SLIP directions\b)"
    ),
    "planes": re.compile(
        r"(expr\s*=\s*')MillerIndex\(torch\.tensor\([^']*\)\)(')(\s*#\s*@SLIP planes\b)"
    ),
}


def _miller(v: Any, what: str) -> Miller:
    """Validate one Miller index: three integers, not all zero."""
    try:
        vals = [int(x) for x in v]
        ok = len(vals) == 3 and all(float(x) == int(x) for x in v)
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f"{what} must be three integers, got {v!r}")
    if not any(vals):
        raise ValueError(f"{what} must not be (0, 0, 0)")
    return (vals[0], vals[1], vals[2])


def _has_perpendicular_variant(d: Miller, p: Miller) -> bool:
    """True if some cubic variant (permutation + signs) of ``d`` is normal to ``p``.

    NEML2 expands a family over the cubic group and keeps the perpendicular
    direction/plane pairs; e.g. ``<110>{111}`` works through ``[1-10].(111) = 0``.
    """
    for perm in itertools.permutations(d):
        for signs in itertools.product((1, -1), repeat=3):
            if sum(s * a * b for s, a, b in zip(signs, perm, p)) == 0:
                return True
    return False


@dataclass(frozen=True)
class SlipSystems:
    """Slip families for a cubic crystal, as written into the NEML2 decks.

    Attributes:
        families: tuple of ``(direction, plane)`` Miller-index pairs; each is
            expanded over the cubic point group by NEML2.
        name: preset name, or ``"custom"`` for user-defined families.
        crystal: crystal class; only ``"cubic"`` is supported.
    """

    families: Tuple[Family, ...]
    name: str = "custom"
    crystal: str = "cubic"

    def __post_init__(self):
        if self.crystal != "cubic":
            raise NotImplementedError(
                f"crystal={self.crystal!r}: only cubic slip systems are supported "
                "(NEML2 v3 exposes CubicCrystal as its only crystal-geometry block)."
            )
        if not self.families:
            raise ValueError("at least one slip family is required")
        fams = []
        for k, fam in enumerate(self.families):
            if isinstance(fam, dict):
                d, p = fam.get("direction"), fam.get("plane")
            else:
                try:
                    d, p = fam
                except (TypeError, ValueError) as err:
                    raise ValueError(
                        f"slip family {k} must be (direction, plane) or "
                        f"{{'direction': ..., 'plane': ...}}, got {fam!r}"
                    ) from err
            d = _miller(d, f"slip family {k} direction")
            p = _miller(p, f"slip family {k} plane")
            if not _has_perpendicular_variant(d, p):
                raise ValueError(
                    f"slip family {k}: no cubic-equivalent of direction {d} lies in "
                    f"plane {p}, so the family has no slip systems"
                )
            fams.append((d, p))
        object.__setattr__(self, "families", tuple(fams))

    @classmethod
    def resolve(
        cls, spec: Union[None, str, "SlipSystems", Iterable[Any]] = None
    ) -> "SlipSystems":
        """Build a :class:`SlipSystems` from a preset name, families, or itself.

        Args:
            spec: ``None`` (the default preset, ``"fcc"``), a preset name from
                :data:`SLIP_SYSTEM_PRESETS`, a :class:`SlipSystems`, or an
                iterable of families, each ``(direction, plane)`` or
                ``{"direction": [...], "plane": [...]}``.

        Returns:
            The validated :class:`SlipSystems`.

        Raises:
            ValueError: unknown preset, malformed family, or a direction not in
                its plane.
        """
        if spec is None:
            spec = DEFAULT_SLIP_SYSTEMS
        if isinstance(spec, SlipSystems):
            return spec
        if isinstance(spec, str):
            key = spec.strip().lower()
            if key not in SLIP_SYSTEM_PRESETS:
                raise ValueError(
                    f"unknown slip-system preset {spec!r}; choose one of "
                    f"{sorted(SLIP_SYSTEM_PRESETS)} or pass (direction, plane) families"
                )
            return cls(SLIP_SYSTEM_PRESETS[key], name=key)
        return cls(tuple(spec))

    @property
    def directions(self) -> Tuple[Miller, ...]:
        """Slip directions, one per family."""
        return tuple(d for d, _ in self.families)

    @property
    def planes(self) -> Tuple[Miller, ...]:
        """Slip-plane normals, one per family."""
        return tuple(p for _, p in self.families)

    def to_dict(self) -> dict:
        """JSON-friendly description (name, crystal, families)."""
        return {
            "name": self.name,
            "crystal": self.crystal,
            "families": [
                {"direction": list(d), "plane": list(p)} for d, p in self.families
            ],
        }

    @staticmethod
    def _tensor_expr(rows: Tuple[Miller, ...]) -> str:
        body = ", ".join("[" + ", ".join(str(x) for x in r) + "]" for r in rows)
        return f"MillerIndex(torch.tensor([{body}], dtype=torch.int64))"

    def apply_to_deck(self, text: str) -> str:
        """Rewrite the ``# @SLIP`` lines of a NEML2 deck with these families.

        Args:
            text: deck source containing exactly one ``# @SLIP directions`` and
                one ``# @SLIP planes`` ``MillerIndex`` line.

        Returns:
            The deck text with both lines replaced.

        Raises:
            RuntimeError: if a marker is missing or duplicated.
        """
        for key, rows in (("directions", self.directions), ("planes", self.planes)):
            expr = self._tensor_expr(rows)
            text, n = _MARKER_RE[key].subn(
                lambda m, e=expr: m.group(1) + e + m.group(2) + m.group(3), text
            )
            if n != 1:
                raise RuntimeError(
                    f"expected exactly one '# @SLIP {key}' MillerIndex line in the "
                    f"NEML2 deck, found {n}"
                )
        return text
