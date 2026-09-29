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
    ("C24", "2026-09-28", "Selection and direction accuracy, not more aggression; 80% confidence gate; 8/10 positive, losers capped",
     "if you ever fall short of a 700% year that doesnt neccessarily mean we arent being aggreeisve enough, it means we arent correctly finding the highly volatile stocks, and then picking out which ones will be positive and which will be negative at a high success rate, if there are external factors you cant controll that cause success chances to be below 80% even with all your tools of checking indicators, such as fda results, no indicator can predict those, and if the chances of its success is below 80% with your tools theres nothing you can do so dont bet on it because the tools you have at your disposal wont accurately enough help you to know if it will be a successful stock, so you need to find 10 stocks that you think will be more volatile that 10% and you need to bet 8/10 will be positive 10% and you need to ensure the remaining to never loose more than 20% and hopefully dont even loose 15 or even 10%"),
    ("C25", "2026-09-28", "Run many experiments; use all the memory all the time",
     "you should be running many experiments at a time\n"
     "i want youto be utilizing all the memory all the time"),
    ("C26", "2026-09-28", "Free memory between projects",
     "when one project ends close everything you can to clear up memory then start a new one"),
    ("C27", "2026-09-28", "Per-stock-type learned indicators and triggers (after picking the right stocks)",
     "the indicators you should be watching and the triggers should be custom to the stock you should have a learning system on what is the most trusting for what type of stock and not just based on the sector, but the size, the current cultural trend ect for it, it should be very detailed, but first you need to ensure you are watching the right stocks before you get to invested in indicators, but if useful and if patterens helps you can also use indicators for helping you find what stocks to pick"),
    ("C28", "2026-09-28", "Three projects: live (little attention), test, finding volatility",
     "right now there should be three projects, live, test, and finding volatility, then the last one will change to consistently finding which ones will be positive, we should give little attention to the live, and instead give our attention to improving the self learning on test and correctly finding the volatile stocks"),
    ("C29", "2026-09-28", "Continuous heavy-test / find-pattern / change-code cycle",
     "we should be constantly adjusting the code to make the testing and the finding volatile stocks work better to improve those numbers running through a heavy test, find pattern, change code cycle"),
    ("C30", "2026-09-28", "Test: three-way pattern study (picked+won, picked+lost, missed winners)",
     "talking to the testing task: find the pattern in what succeeded that the system picked, what failed that the system picked, and what succeeded that the system didnt pick, and then identify the thing that needs to be motified to get us closer to our goal, feel free to incoperate the find volatility system to find volatile stocks, but that still isnt complete and you still need to find patterns in the indicators, volume, quantity, net worth ect"),
    ("C31", "2026-09-28", "Test objective: first 7%, second accuracy",
     "the test should keep adjusting first keeping in mind 7% then second keeping in mind accuracy of success"),
    ("C32", "2026-09-28", "Checkoff list; any method or free tool; all RAM; don't stop until done",
     "treat this as a checkoff list and feel free to motify anything or look at anything in a new way or download any free tools to achieve the ultimate objective but tread this as a checkoff list and utilize all the RAM and dont stop until the list is done"),
    ("C33", "2026-09-28", "No cheating; never after-market or weekend trading",
     "ensure there is both no  cheating and never any after market trading or weekend trading"),
    ("C34", "2026-09-28", "Advanced self-learning with advanced, factor-weighted memory",
     "we need a algorithm for advanced self learning system, but we also need the algorithm to have advanced memory knowing how prevelant to make memory based on a variety of factors"),
    ("C35", "2026-09-28", "Algorithm project: self-learning pattern system with massive, relevance-weighted memory",
     'memory should always be stored but more recent things, more common trends, more modern fields, more modern situations such as war, effect stuff like that, it should be self learning no just to always learn more to get infinitiely smart over a period of time but figure out how to make it so its best customized to when its running through what memory it says is most important, and what facts are consistent and the self learning should pick up minor details it should pick up the big obvious details, but also the very small changes only a computer would notice like small reversals, momentary reversals, patterns between different time line candles from minute candles to monthly candles, it should pick up thousands of patterns finding even the smallest thing like if something only effects... unless ... then it cancels out, some of the data will be impossible to notice without an extreme system testing pool and thats why we need you to create a massive memory, but to know what memory should be used when, because data from 1962 will probably be rarely relevent but it still could be sometimes, you need to generate it so its not just looking at the patterns from new test results but also so it looks back at its memory to develop patterns getting extrodinarily advanced and doing statistical math to calculate chance of coincidence and chance of it having an effect, finding how big the effect is and adjusting for the size of the effect and if patterns begin to die, they should look for reasons why they died and try to develop a new pattern but even if no new pattern develops a not working pattern should not be used unless you can find the sourse of why its not working, this isnt for you to do though, its for you to build the self learning system that can build a algorithm like this, so now we have test, find volatility, live, and algorithm, everytime test results come back you should review all three tasks that it applies to going through every single one, keeping in mind that we need a lot of tests, so while you work have tests working, starting a new test should be the first thing to do before you make all the changes so you can work and wait for the test simultaniously, dont stop until this is done and save this to your memory file verbatum'),
    ("C36", "2026-09-28", "Priority: Algorithm, then Find volatility, then minimal direct Test changes",
     "highest priority is algorithm, find volatility and direct changes to test happen below that but we priortize the learning sytem above direct changes to test, only change what is necessarry for task test because algorithm should be able to do most of it"),
    ("C37", "2026-09-28", 'Algorithm blueprint first: microscopic effects; P(real / hallucinated / coincidence); thousands of patterns; never stops, never learns for no reason',
     'make the bluprint first and structure it aropund the microscopic effects only a computer would notice the calculate the chances of statistical likelyness of the noticed pattern being real, hallucinated, or coincidence, and structure it around being able to learn thousands of patterns and it never stops trying to learn, but also it never learns for no reason'),
    ("C38", "2026-09-28", '7% weekly is the goal: below is too low, above is too risky; keep feeding it more data',
     'also make sure that the self learning keeps in mind 7% weekly is the goal so anything below that should be seen as to low and anything above it should be seen as to risky, and you should spend time just giving it more data while you build it so the self learning has more to work with'),
    ("C39", "2026-09-28", 'Priority: 7% weekly volatility first, then minimise risk, then raise the share of positive 7% weeks',
     'it should minimize risk as much as possible but 7 % weekly is top priority, try to minimize risk after you achieve that, that should be a fundamental built into the system where as long as most weeks are 7% volatility then the top priority has been met, after that you can increase the percentage of the 7% being positive rather than negative'),
    ("C40", "2026-09-28", 'Master blueprint, in Downloads, very long',
     'make me a master blueprint\nput it in file downloads\nit should be very long'),
    ("C41", "2026-09-28", 'Rare-analog memory: notice when today resembles the one other time something happened; every data point knows its date and relevance',
     'the algorithm should be so advanced that if there was only one other time in history that a certain stock market problem had ever occured like a bubble pop or a sector disapear or a sector arise, or even something very small most humans would never find it but a small thing only happened one other time in history, the self learning should notice the simalarity, the self learning should also know what year all data points occured to know what coordinates with what, and how relavent data points are to today'),
    ("C42", "2026-09-28", '7% average over a year; timeline-aware caution/aggression without cheating; learn from mistakes so a rerun would not repeat them (by pattern, not by memorised answers)',
     "it should be so advanced that it aims for the average of 7% a week over the course of a year, but it also notices the timelines to be more cautious or more aggressive without cheating by investigating what timeline it is, and if it messes up on a test, then it should self learn so if we run the same test again it wouldn't mess up the second time, in fact you should give that a try where it doesnt remember oh yeah now i need to pick this stock, but oh yeah, i noticed a time where this pattern caused this to happen so im going to try something else instead"),
    ("C43", "2026-09-28", 'Never hold a failed pattern: improve it until consistent up to that point, or discard it',
     'we should never hold onto a pattern that failed, the pattern either needs to be improved to something that would be consistent up to that point or disguarded'),
    ("C44", "2026-09-28", "Master blueprint covers every detail, including undiscussed and uncertain ones",
     'make sure the master blueprint literally covers every detail from every single thing the system looks at to what should be built in to what the system can learn on its own, to how risky to be ect every single little detail in the master blueprint, even the details we havent discussed that you know the answer to or even things you dont know 100% how it will work yet'),
    ("C45", "2026-09-28", 'The Bible: Master Autonomous Build & Validation Prompt v1.0 (verbatim in BIBLE.md, sha256 in canon/bible.lock.json)',
     'save this as your master prompt task to follow: [WEEKLY7 MASTER AUTONOMOUS BUILD & VALIDATION PROMPT v1.0, stored verbatim in BIBLE.md] or just call it your bible'),
    ("C46", "2026-09-28", 'Line-count rule',
     'im also creating a rule where if you do not have the number of code lines in the expected range its not detailed enough so correct it to be more detailed'),
    ("C47", "2026-09-28", 'Check every box, in the best order, with tests running',
     'check off the master list in whatever order you see best just make sure as you do it you are running tests to collect data, and that all the checkboxes get checked before you call it'),
    ("C48", "2026-09-28", 'Keep coding continuously; tests run in the background and are reviewed later',
     'as you work on this you can have things running but I want you consistently coding cause theres a lot of code to get through and you can go back and test everything later but dont stop coding no matter what'),
    ("C49", "2026-09-28", 'Save everything learned (from the owner or from the system) to memory files',
     'also ensure anything you learn from me or yourself gets saved into memory files to ensure we dont loose any valuable stuff'),
    ("C50", "2026-09-28", 'Foundation before rabbit holes: 25,000 lines minimum first',
     'we need our foundation before we need rabit holes so dont dive into any rabit holes until you have the 25,000 minimum lines of code'),
    ("C51", "2026-09-28", '25,000 is a floor, not a cap; every foundation before any rabbit hole',
     'you can go over 25,000 though just make sure everythings foundation is done before you touch anythings rabit hole'),
    ("C52", "2026-09-28", 'Masterstock: one file anyone can read to fully catch up',
     'save everything at all that happens in a single large file so that i can switch claude accounts and still just point them to one file for them to fully catch up, call the file/folder Masterstock'),
    ("C53", "2026-09-28", 'Handoff files current at any moment (session switch may come without notice)',
     'make sure my files are fully updated because i may need to switch claude sessions sometime soon so that needs to stay up to date for any given moment'),
    ("C54", "2026-09-29", 'The learning system is the product: judge a test by how results CHANGE when the same year is run again',
     'remember you arent as focused on the system as you are the learning system meaning when a test comes back its less about the results you see that time, but the changes in the results the second time you run the same year'),
    ("C55", "2026-09-29", 'The rerun of a year must be disguised: the system must not know it is the same year',
     '(it should not be aware its the same year however)'),
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
