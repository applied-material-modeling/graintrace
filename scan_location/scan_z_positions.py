import csv
import io


def compute_scan_positions(z_ref, window_height, overlap, total=1000):
    """
    Compute z scan positions.

    Args:
        z_ref: reference z position (um)
        window_height: scan window height (um)
        overlap: overlap fraction (0.0 to 1.0)
        total: total scan range (um), default 1000

    Returns:
        list of absolute z positions
    """
    step = window_height * (1 - overlap)
    positions = []
    z = window_height / 2  # center of first window, so bottom edge = z_ref
    while True:
        positions.append(z_ref + z)
        if z + window_height / 2 >= total:  # top edge >= total above z_ref
            break
        z += step
    return positions


def main():
    z_ref = float(input("Enter z_ref (um): "))
    total = float(input("Enter total scan range (um): "))

    cases = [
        (200, 0.00),
        (100, 0.00),
        (50, 0.00),
        (25, 0.00),
        (200, 0.25),
        (200, 0.50),
        (100, 0.25),
        (100, 0.50),
        (10, 0.00),
    ]

    output_lines = []
    output_lines.append(f"\n{'='*70}")
    output_lines.append(f"  z_ref = {z_ref} um    |    Total scan range = {total} um")
    output_lines.append(f"{'='*70}")

    csv_rows = []  # each row: case label + z positions
    summary_rows = []  # simple summary: case, window_size, first_position, step

    for i, (window_height, overlap) in enumerate(cases, 1):
        positions = compute_scan_positions(z_ref, window_height, overlap, total)
        step = window_height * (1 - overlap)
        z_min = positions[0] - window_height / 2  # bottom of first window = z_ref
        z_max = positions[-1] + window_height / 2
        label = f"W{window_height}_O{int(overlap*100)}pct"
        output_lines.append(
            f"\n  Case {i:>2}: Window={window_height:>3} um | Overlap={overlap*100:>4.0f}% | Step={step:>6.1f} um | z_min={z_min} um | z_max={z_max} um | N={len(positions)}"
        )
        output_lines.append(f"          z locations: {positions}")
        csv_rows.append([label] + positions)
        summary_rows.append([i, window_height, positions[0], step, len(positions)])

    full_output = "\n".join(output_lines)
    print(full_output)

    txt_path = f"scan_z_positions_zref{int(z_ref)}_total{int(total)}.txt"
    with open(txt_path, "w") as f:
        f.write(full_output + "\n")
    print(f"\n  Saved text output -> {txt_path}")

    csv_path = f"scan_z_positions_zref{int(z_ref)}_total{int(total)}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "z_locations_um (centers)"])
        max_len = max(len(r) - 1 for r in csv_rows)
        writer.writerow(["case"] + [f"z{j+1}" for j in range(max_len)])
        for row in csv_rows:
            writer.writerow(row)
    print(f"  Saved CSV         -> {csv_path}")

    summary_path = f"scan_z_summary_zref{int(z_ref)}_total{int(total)}.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case",
                "window_size_um",
                "first_position_um",
                "step_size_um",
                "total_scans",
            ]
        )
        for row in summary_rows:
            writer.writerow(row)
    print(f"  Saved summary CSV -> {summary_path}")

    total_all = sum(row[4] for row in summary_rows)
    print(f"\n{'='*70}")
    print(f"  Total scans per case:")
    for row in summary_rows:
        print(
            f"    Case {row[0]:>2}: W={row[1]:>3} um | step={row[3]:>6.1f} um -> {row[4]} scans"
        )
    print(f"  {'─'*50}")
    total_excl_last = sum(row[4] for row in summary_rows[:-1])
    print(f"  Grand total across all cases: {total_all} scans")
    print(f"  Grand total excluding last case: {total_excl_last} scans")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
