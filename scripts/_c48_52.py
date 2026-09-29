"""Append canon C48-C52 (owner directives of 28 Sep 2026, verbatim) to canon/build_canon.py."""
import pathlib
D = [("C48", "Keep coding continuously; tests run in the background and are reviewed later",
      "as you work on this you can have things running but I want you consistently coding cause theres a lot of code to get through and you can go back and test everything later but dont stop coding no matter what"),
     ("C49", "Save everything learned (from the owner or from the system) to memory files",
      "also ensure anything you learn from me or yourself gets saved into memory files to ensure we dont loose any valuable stuff"),
     ("C50", "Foundation before rabbit holes: 25,000 lines minimum first",
      "we need our foundation before we need rabit holes so dont dive into any rabit holes until you have the 25,000 minimum lines of code"),
     ("C51", "25,000 is a floor, not a cap; every foundation before any rabbit hole",
      "you can go over 25,000 though just make sure everythings foundation is done before you touch anythings rabit hole"),
     ("C52", "Masterstock: one file anyone can read to fully catch up",
      "save everything at all that happens in a single large file so that i can switch claude accounts and still just point them to one file for them to fully catch up, call the file/folder Masterstock")]
p = pathlib.Path("canon/build_canon.py"); s = p.read_text(encoding="utf-8")
assert '("C48"' not in s, "already added"
end = s.index("\n]\n", s.index('("C47"'))
s = s[:end + 1] + "".join(f'    ("{c}", "2026-09-28", {t!r},\n     {x!r}),\n' for c, t, x in D) + s[end + 1:]
compile(s, "build_canon.py", "exec")
p.write_text(s, encoding="utf-8", newline="\n")
print("added", [d[0] for d in D])
