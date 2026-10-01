from app.textutil import title_case

def test_title_case():
    assert title_case("hello wORLD") == "Hello World"
