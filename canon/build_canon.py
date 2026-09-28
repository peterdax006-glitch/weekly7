"""Regenerates canon/CANON.md from the verbatim directives below.
Directives are append-only and stored exactly as the user typed them (typos included).
Each carries a SHA-256 of its UTF-8 text so any alteration is detectable: run with --verify."""
import hashlib, json, pathlib, sys

HERE = pathlib.Path(__file__).parent
DIRECTIVES = [
    ("C1", "2026-09-28", "Mission",
     "I want you to build a website that does one thing, it picks stocks, it should be very advanced and all the apis or anything like that used should be free, you should aim for 7% a week and it can update investments however best you see fit if thats everyday, if thats live changing constantly or no matter what it is, you can pick as many stocks as you want but it should aim for 7% gains every week, this will only be used as a simulation it will have the account start with $1000 and you can invest that however you want and over the next month we will moniter how successfull it is, but first I want you to do deep research and create a master blueprint to make the best stock finder possible, that doesn't just follow trending stocks, but the ones that actually will grow 7% in a week"),
    ("C2", "2026-09-28", "Target is portfolio-level",
     "this means the entire portfolio should grow 7%, not neccessarily every stock choic"),
    ("C3", "2026-09-28", "Blueprint must be advanced",
     "the blueprint should be very advanced, seeing everything that creates the most accurate stradegy to the expected outcome"),
    ("C4", "2026-09-28", "Permission to set up accounts",
     "i give you permission to do all that in claude in code including making github and all that, i can get you logged into anything you need"),
    ("C5", "2026-09-28", "Continuous self-improvement",
     "it should have an advanced learning process like any ai of figuring out what isnt working and how to improve the system to fix that thing that wasnt working until we land the perfect system add this to memory files and a canon to work on"),
    ("C6", "2026-09-28", "Regular trading hours only",
     "also it should onl y trade durring daytime trading hours(continue working on what you were doing just keep that in mind"),
    ("C7", "2026-09-28", "Hundreds of historical simulations",
     "i want you to run simulations using past data where it finds the perfect stock from a random year from a random decade using the system, and then you anaylize how it did, they you adjust, you should run hundreds of tests to get it perfect."),
    ("C8", "2026-09-28", "No cheating in tests; keep the weekly 7% focus",
     "it should not cheat though when finding random years from random decades, it should be fully oblivious to what will happen to the stock so it doesnt just choose the best ones, and it should still focus on 7% over a weekly basis even in the tests"),
    ("C9", "2026-09-28", "Replay a random past year as if live",
     "on the past one i want you to run it on your system, picking a random year being sure not to cheat and doing every week in a year, updating the portfolio however often you want, but it should only trade durring trading days, and it should treat the past as if its live, but time is just moving way way way faster so the system is working very fast"),
    ("C10", "2026-09-28", "Loop random years until 7%/week average",
     "do that in random years until you average 7% a week, adjusting the system as neccessarry, being sure to pick a random year over the last 50 years everytime"),
    ("C11", "2026-09-28", "Blind live-clock simulation of a random hidden year; no crypto",
     "no sorry i dont want you to test every year, i want you to test a random year between 1962-today, where not even your system knows the year its playing with, and the system speed runs through the entire year it runs through a live simulation where there is actually a clock going very fast and your system picks stocks live based on your system, and then at the end you examine things that made your stocks not as high and see what can be adjusted, then run the simulation again, continue that process until you hit 7% a week\n"
     "the clock should be ticking as fast as where your system can just keep up at a live pase to make sure we dont have it be a second slower than possible\n"
     "also crypto is off limits\n"
     "continue working while you wait on that"),
    ("C12", "2026-09-28", "Make testing very fast before bulk testing",
     "ensure you get testing to go very fast so you can run tests very fast, learn how you can speed up testing before you start bulk testing to speed up the progress"),
    ("C13", "2026-09-28", "Pattern explorer across all indicators",
     "make sure there is a way you can view the exact patterns of stocks that got us closer to our goal looking for patters in all the dozens of different trends and indicators and quantity of everything"),
    ("C14", "2026-09-28", "Sensitivity: what moves the outcome, how much, how consistently",
     "figure out what things adjust the outcome significantly and what things adjust it slightly, and what things are more consistent and what are less consistent, and how much to adjust different indicators and any other trackers you use and play with it to see in a real scenario how much it actually effects anything so then for the next loop you know what to adjust to hopefully make the weekly average significantly higher consistently"),
    ("C15", "2026-09-28", "Self-adjusting system within a year",
     "also have a way for the system to automatically adjust itself, so if this year... is working better then slightly motify the system within a test, but also keep in mind that a major drop off in that trend could happen randomly depending on... depending on the indicator"),
    ("C16", "2026-09-28", "No manual adjustment mid-test; train the training basis each loop",
     "so as you run a test you shouldn't adjust things, but design it for a self training system inside of a year, adjusting on a weekly basis based on whats working, but everytime we run a loop you will need to train the traing basis better, especially for things like making sure it works well enough and compensates for coincidents, or things you wouldnt normally expect, or not enough data to go on ect"),
    ("C17", "2026-09-28", "Pre-season self-training of defaults",
     "also at the begining of a year when you run a test it should research and use tools to figure train itself what the default should be at the start of the year then have it motify itself as the year progresses"),
    ("C18", "2026-09-28", "No way to cheat in the self-adjusting design",
     "be very cautious the system doesnt have a way to cheat as you design this though"),
    ("C19", "2026-09-28", "Random start month, 12 consecutive months",
     "also it should not be set up so you go from the start of the year to the end of it, testing should pick a random month over the last however many years, and have it run for the next 12 consecutive months"),
    ("C20", "2026-09-28", "Learn from the winners it did not pick",
     "it should also train itself, not just by its picks, but the successes it didnt pick, and look at patterns with those and see if there was any way to predict it, so then its trained to get it next time, so it should examine maybe hundreds on non picks every year too"),
    ("C21", "2026-09-28", "Below 1%: raise volatility toward +/-5% first",
     "also if you stay below 1% then just make the changes to make it more volatile to get closer to that 5% negative or positive and then figure out consistent positives later"),
    ("C22", "2026-09-28", "Much more volatile: +/-200% a year",
     "this result is not nearly volatile enough we want to be seeing 200%-200% returns over a year"),
    ("C23", "2026-09-28", "Weekly 10 movers: 95% then direction, exit, stop; +10% 8x as often as -10%",
     "okay what we want to do seperately while also doing all this, is every week we want it to find 10 stocks that it thinks will move 10% up or down from its current position in the next week, it should give a percentage of how often its correct, you should adjust it until its correct 95% of the time, once that happens it should look for patterns to tell which ones are going to go up or down, exaclty when to sell, and exactly where to have a stop loss, the goal is to get postive 10% 8 times more than a negative 10% and you should play with it until you can consistently achieve that no matter the year, without cheating, make sure to do everything in the order i said and dont skip steps"),
]

def sha(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()

def build():
    lines = ["# Weekly7 Canon", "",
             "The user's directives, **verbatim** and append-only. They outrank the blueprint; the blueprint outranks the code.",
             "Generated by `canon/build_canon.py` — edit the script, never this file. Verify: `python canon/build_canon.py --verify`.", ""]
    for cid, date, title, text in DIRECTIVES:
        lines += [f"## {cid} — {title} ({date})", "", "> " + text, "", f"`sha256: {sha(text)}`", ""]
    (HERE / "CANON.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (HERE / "canon.lock.json").write_text(json.dumps({c: sha(t) for c, _, _, t in DIRECTIVES}, indent=2), encoding="utf-8", newline="\n")

def verify():
    lock = json.loads((HERE / "canon.lock.json").read_text(encoding="utf-8"))
    bad = [c for c, _, _, t in DIRECTIVES if lock.get(c) != sha(t)]
    missing = [c for c in lock if c not in {d[0] for d in DIRECTIVES}]
    if bad or missing:
        print("CANON ALTERED:", bad, "MISSING:", missing); sys.exit(1)
    print(f"canon ok: {len(lock)} directives")

if __name__ == "__main__":
    verify() if "--verify" in sys.argv else build()
