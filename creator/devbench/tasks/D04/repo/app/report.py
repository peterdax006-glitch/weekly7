def sales_report(rows):
    out = []
    for name, value in rows:
        out.append(f"{name:<10}|{value:>8.2f}")
    return "\n".join(out)


def cost_report(rows):
    lines = []
    for name, value in rows:
        lines.append(f"{name:<10}|{value:>8.2f}")
    return "\n".join(["COSTS"] + lines)
