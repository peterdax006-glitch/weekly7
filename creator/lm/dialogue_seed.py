"""The teacher's first 60 lessons (stages 2-4): 40 train, 20 held-out. Turns alternate user, nupen, ... and end on the reference Nupen reply.

Tuple = (stage, split, entities, expect, turns). `entities` are the names/facts that must never cross the train/heldout split;
`expect` are the acceptable answer words for the final reply (None = stage 2, only turn-taking is scored).
"""
from __future__ import annotations

from typing import Any

T, H = "train", "heldout"

_SEED: list[tuple[int, str, tuple[str, ...], tuple[str, ...] | None, tuple[str, ...]]] = [
    # ---- stage 2: turn-taking (14 train, 6 heldout) ----
    (2, T, (), None, ("Hello!", "Hello! How can I help you today?")),
    (2, T, (), None, ("Good morning.", "Good morning to you too.")),
    (2, T, (), None, ("How are you?", "I am well, thank you. How are you?", "I am fine.", "I am glad to hear that.")),
    (2, T, (), None, ("What is your name?", "My name is Nupen.")),
    (2, T, (), None, ("Thank you.", "You are welcome.")),
    (2, T, (), None, ("Are you there?", "Yes, I am here.")),
    (2, T, (), None, ("Please say hello.", "Hello.")),
    (2, T, (), None, ("I like music.", "Music is nice. What kind do you like?", "I like piano.", "Piano is a lovely sound.")),
    (2, T, (), None, ("Goodbye.", "Goodbye. Have a good day.")),
    (2, T, (), None, ("It is raining today.", "Then take an umbrella.")),
    (2, T, (), None, ("I am tired.", "Then you should rest.")),
    (2, T, (), None, ("Can you help me?", "Yes. What do you need?")),
    (2, T, (), None, ("Tell me something.", "The sun is a star.")),
    (2, T, ("soup",), None, ("I made soup.", "That sounds warm and good.")),
    (2, H, (), None, ("Hi there.", "Hi! It is nice to talk with you.")),
    (2, H, (), None, ("Good evening.", "Good evening. I hope your day went well.")),
    (2, H, (), None, ("Who are you?", "I am Nupen. I am learning to talk.")),
    (2, H, (), None, ("Thanks a lot.", "No problem at all.")),
    (2, H, ("sandwich",), None, ("I am hungry.", "Then you should eat something.", "I will make a sandwich.", "That is a good idea.")),
    (2, H, (), None, ("See you later.", "See you later. Take care.")),
    # ---- stage 3: answer from the conversation (14 train, 7 heldout) ----
    (3, T, ("Anna", "hat"), ("anna",), ("My name is Anna. I have a red hat.", "Nice to meet you, Anna.", "What is my name?", "Your name is Anna.")),
    (3, T, ("Ben", "dog", "Max"), ("max",), ("Ben has a dog called Max.", "Max is a nice name.", "What is the dog called?", "The dog is called Max.")),
    (3, T, ("Carlos", "Lima"), ("lima",), ("Carlos lives in Lima.", "Lima is a big city.", "Where does Carlos live?", "Carlos lives in Lima.")),
    (3, T, ("Dina", "bag"), ("blue",), ("Dina has a blue bag.", "That sounds useful.", "What colour is the bag?", "The bag is blue.")),
    (3, T, ("Eli", "cups"), ("four", "4"), ("Eli put four cups on the table.", "That is a lot of cups.", "How many cups are on the table?", "There are four cups.")),
    (3, T, ("Fatima", "rice"), ("rice",), ("Fatima likes rice.", "Rice is a good meal.", "What does Fatima like?", "Fatima likes rice.")),
    (3, T, ("Greg", "horse"), ("greg",), ("My horse is brown and my friend Greg has a grey one.", "Two horses, then.", "Who has the grey horse?", "Greg has the grey horse.")),
    (3, T, ("Hana", "Oslo", "Cairo"), ("cairo",), ("Hana went to Oslo on Monday and to Cairo on Friday.", "That is a long trip.", "Where did Hana go on Friday?", "Hana went to Cairo on Friday.")),
    (3, T, ("Ivan", "ball"), ("green",), ("Ivan kicked a green ball over the fence.", "Oh no.", "What did Ivan kick?", "Ivan kicked a green ball.")),
    (3, T, ("Jade", "frog"), ("jade",), ("Jade found a small frog in the garden.", "Frogs like gardens.", "Who found the frog?", "Jade found the frog.")),
    (3, T, ("Kofi", "bread"), ("seven", "7"), ("Kofi baked seven loaves of bread.", "That is a busy day.", "How many loaves did Kofi bake?", "Kofi baked seven loaves.")),
    (3, T, ("Lena", "duck"), ("yellow",), ("Lena drew a yellow duck.", "Ducks are fun to draw.", "What colour is the duck?", "The duck is yellow.")),
    (3, T, ("Marco", "Rome", "book"), ("book",), ("Marco bought a book in Rome.", "Was it a good book?", "What did Marco buy?", "Marco bought a book.")),
    (3, T, ("Nora", "rabbit"), ("white",), ("My name is Nora and I have a white rabbit.", "A white rabbit sounds lovely.", "What colour is my rabbit?", "Your rabbit is white.")),
    (3, H, ("Zed", "kite"), ("red",), ("Zed has a red kite.", "That sounds fun.", "What colour is the kite?", "The kite is red.")),
    (3, H, ("Yara", "Madrid"), ("madrid",), ("Yara lives in Madrid.", "Madrid is far from here.", "Where does Yara live?", "Yara lives in Madrid.")),
    (3, H, ("Quinn", "lamp"), ("three", "3"), ("Quinn bought three lamps.", "That is a bright idea.", "How many lamps did Quinn buy?", "Quinn bought three lamps.")),
    (3, H, ("Wanda", "turtle"), ("wanda",), ("Wanda has a slow turtle.", "Turtles are calm.", "Who has the turtle?", "Wanda has the turtle.")),
    (3, H, ("Victor", "pear"), ("pear",), ("Victor eats a pear every day.", "That is healthy.", "What does Victor eat every day?", "Victor eats a pear.")),
    (3, H, ("Xavier", "Nairobi", "Prague"), ("prague",), ("Xavier flew to Nairobi, and then to Prague.", "He saw a lot.", "Where did Xavier fly last?", "Xavier flew to Prague last.")),
    (3, H, ("Bea", "drum", "goat"), ("goat",), ("Bea plays the drum while her goat listens.", "What a funny picture.", "Who listens to Bea play?", "Her goat listens.")),
    # ---- stage 4: instructions (12 train, 7 heldout) ----
    (4, T, (), ("the sky is high",), ("Repeat after me: the sky is high.", "The sky is high.")),
    (4, T, (), ("hello",), ("Say hello.", "Hello.")),
    (4, T, (), ("5",), ("What is 2+3?", "2+3 is 5.")),
    (4, T, (), ("9",), ("What is 4+5?", "4+5 is 9.")),
    (4, T, ("fire",), ("yes",), ("Say yes or no: is fire hot?", "Yes.")),
    (4, T, ("stone",), ("no",), ("Say yes or no: is a stone alive?", "No.")),
    (4, T, ("water",), ("yes",), ("Answer yes or no. Is water wet?", "Yes.")),
    (4, T, (), ("tuesday",), ("What day comes after Monday?", "Tuesday comes after Monday.")),
    (4, T, (), ("i like to read",), ("Repeat after me: I like to read.", "I like to read.")),
    (4, T, (), ("7",), ("What is 3+4?", "3+4 is 7.")),
    (4, T, (), ("sam",), ("Say hello to Sam.", "Hello, Sam.")),
    (4, T, (), ("one", "two", "three"), ("Count from one to three.", "One, two, three.")),
    (4, H, (), ("birds can sing",), ("Repeat after me: birds can sing.", "Birds can sing.")),
    (4, H, (), ("8",), ("What is 10 minus 2?", "10 minus 2 is 8.")),
    (4, H, ("ice",), ("yes",), ("Say yes or no: is ice cold?", "Yes.")),
    (4, H, ("moon",), ("no",), ("Say yes or no: is the moon a fish?", "No.")),
    (4, H, (), ("wednesday",), ("What day comes after Tuesday?", "Wednesday comes after Tuesday.")),
    (4, H, (), ("the tea is hot",), ("Repeat after me: the tea is hot.", "The tea is hot.")),
    (4, H, ("Rita",), ("bye",), ("Say bye to Rita.", "Bye, Rita.")),
]


def seed_items() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for stage, split, ents, expect, turns in _SEED:
        dlg = [{"role": "user" if i % 2 == 0 else "nupen", "text": t} for i, t in enumerate(turns)]
        item: dict[str, Any] = {"stage": stage, "dialogue": dlg, "split": split, "source": "teacher", "entities": list(ents)}
        if expect is not None:
            item["expect"] = list(expect)
        out.append(item)
    return out
