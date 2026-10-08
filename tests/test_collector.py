from bs4 import BeautifulSoup

from data_collection.collector import AirBankCollector


def test_article_url_filter() -> None:
    assert AirBankCollector._is_article_url(
        "https://www.airbank.cz/co-vas-nejvic-zajima/sporici-ucet/"
    )
    assert not AirBankCollector._is_article_url(
        "https://www.airbank.cz/co-vas-nejvic-zajima/rubrika/bezny-a-sporici-ucet/"
    )
    assert not AirBankCollector._is_article_url("https://example.com/co-vas-nejvic-zajima/foo/")


def test_extract_html_text_removes_navigation_and_related_topics() -> None:
    soup = BeautifulSoup(
        """
        <html><body>
          <header><p>Header noise</p></header>
          <main>
            <h1>Spořicí účet</h1>
            <p>První důležitá informace.</p>
            <ul><li>Druhá důležitá informace.</li></ul>
            <h2>Další témata</h2>
            <p>Toto je související navigace, ne článek.</p>
          </main>
          <footer><p>Footer noise</p></footer>
        </body></html>
        """,
        "html.parser",
    )

    text = AirBankCollector._extract_html_text(soup)

    assert "Spořicí účet" in text
    assert "První důležitá informace." in text
    assert "Druhá důležitá informace." in text
    assert "Další témata" not in text
    assert "související navigace" not in text
    assert "Header noise" not in text
    assert "Footer noise" not in text


def test_category_cleaning_removes_more_information_suffix() -> None:
    assert AirBankCollector._clean_category("Naše účty Více informací") == "Naše účty"
