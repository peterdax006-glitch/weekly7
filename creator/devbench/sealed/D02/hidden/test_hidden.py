from app.textutil import slugify, title_case

def test_slugify():
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  --Already--Sluggy  ") == "already-sluggy"
    assert slugify("A1 b2___c3") == "a1-b2-c3"
    assert slugify("!!!") == ""

def test_title_case_kept():
    assert title_case("a b") == "A B"
