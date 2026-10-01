def format_row(name, value):
    return f"{name:<10}|{value:>8.2f}"


def sales_report(rows):
    return "\n".join(format_row(n, v) for n, v in rows)


def cost_report(rows):
    return "\n".join(["COSTS"] + [format_row(n, v) for n, v in rows])
