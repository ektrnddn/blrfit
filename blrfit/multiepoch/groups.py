"""The behavior groups of the robust movers. The groups follow from the tags of ``verdicts`` by fixed rules, in this
order, so the grouping is reproducible:

  steady    a line drifts steadily (a straight line in time fits) at a rate its virial mass allows: what a binary seen
            over a short arc of a long orbit does
  too_fast  every moving line changes faster than an orbit at its virial mass can: not orbital motion at that mass
            (or the mass is badly wrong)
  both      both Balmer lines move, together, but not steadily: a real, coherent change of the broad-line region,
            zig-zagging over the years
  one       one line moves, not steadily; the other is unmeasured or unchanged within its larger errors
"""

GROUPS = {
    "steady": dict(
        title="A steady drift the mass allows",
        text="One line drifts at a steady rate (a straight line in time fits its velocities) and no change is faster than an "
        "orbit at its virial mass allows. This is what a binary seen over a short arc of a long orbit does.",
    ),
    "too_fast": dict(
        title="Faster than the mass allows",
        text="Every moving line changes faster than any orbit at its virial mass can make it change, so the motion is not "
        "orbital at that mass (or the mass is badly wrong).",
    ),
    "both": dict(
        title="Both lines move, not steadily",
        text="Both Balmer lines move, and together, but their velocities go up and down rather than drifting steadily: a "
        "coherent change of the broad-line region over the years.",
    ),
    "one": dict(
        title="One line moves, not steadily",
        text="One line moves and zig-zags; the other is unmeasured or unchanged within its larger errors.",
    ),
}
ORDER = ("steady", "both", "too_fast", "one")


def group_of(o, vd):
    """vd: (object, line) -> verdicts.csv row. Returns the group key of a robust mover."""
    mv = [vd[(o, ln)] for ln in ("Halpha", "Hbeta") if (o, ln) in vd and vd[(o, ln)]["verdict"] == "moving"]
    tags = [v["tags"] for v in mv]
    steady_ok = any((";steady" in ";" + t) and ("faster than" not in t) for t in tags)
    if steady_ok:
        return "steady"
    if tags and all("faster than" in t for t in tags):
        return "too_fast"
    if len(mv) == 2:
        return "both"
    return "one"


# ------------------------------------------------------------------ groups by the orbit test
# ``orbits`` fits every moving line with the circular orbits the virial mass allows (all periods from P_min up, the binary's
# centre of mass within +-300 km/s of the narrow lines); these groups follow that test.
ORBIT_GROUPS = {
    "orbit": dict(
        title="An orbit the mass allows fits",
        text="Every moving line is fitted by a circular orbit the virial mass allows (all periods from the shortest its "
        "broad-line region permits; the centre of mass at the narrow lines): with five or more times at p >= 0.01, "
        "with four or fewer the orbit passes through the points but cannot be tested.",
    ),
    "no_orbit": dict(
        title="No orbit the mass allows fits",
        text="No circular orbit the virial mass allows reaches the velocities of a moving line: the best one costs more "
        "than Delta chi2 = 4 against an orbit without the mass condition, or fits at p < 0.01.",
    ),
}
ORBIT_ORDER = ("orbit", "no_orbit")


def orbit_group_of(o, vd, orbits):
    """vd: (object, line) -> verdicts.csv row; orbits: 'object|line' -> orbits record."""
    mv = [ln for ln in ("Halpha", "Hbeta") if (o, ln) in vd and vd[(o, ln)]["verdict"] == "moving"]
    ok = [orbits.get(f"{o}|{ln}", {}).get("orbit_fits") for ln in mv]
    return "orbit" if ok and all(x is True for x in ok) else "no_orbit"
