"""Mock story data for admin email template preview."""

from datetime import datetime, timedelta

from newsbot.constants import TZ, label_for_score
from newsbot.models import (
    Article,
    SentimentResult,
    SentimentSummary,
    Story,
    StoryAnalysis,
    SummaryItem,
)


def _mock_article(
    title: str,
    source: str,
    url: str,
    summary: str,
    compound: float,
    *,
    pub_date: datetime,
    now: datetime,
) -> Article:
    label = label_for_score(compound)
    sentiment = SentimentResult(
        article_url=url,
        source=source,
        polarity=compound,
        subjectivity=0.5,
        compound=compound,
        label=label,
    )
    return Article(
        title=title,
        content="",
        source=source,
        url=url,
        published_date=pub_date,
        scraped_date=now,
        summary=summary,
        sentiment=sentiment,
    )


def _mock_sentiment(articles: list[Article]) -> SentimentSummary:
    sentiments = [a.sentiment for a in articles if a.sentiment]
    avg = (
        sum(s.compound for s in sentiments) / len(sentiments)
        if sentiments
        else 0.0
    )
    return SentimentSummary(
        avg_sentiment=avg,
        label=label_for_score(avg),
        article_count=len(sentiments),
        sentiments=sentiments,
    )


def _summary_items(*articles: Article) -> list[SummaryItem]:
    return [
        SummaryItem(article=article, summary=article.summary)
        for article in articles
    ]


def build_preview_story_analyses() -> list[StoryAnalysis]:
    """Build mock story analyses for email template preview."""
    now = datetime.now(TZ)
    pub_date = now - timedelta(days=3)

    art1a = _mock_article(
        "Central banks signal coordinated rate pause after inflation data",
        "Financial Times",
        "https://example.com/1a",
        (
            "Major central banks including the Fed and ECB signaled a "
            "coordinated pause on rate hikes after inflation figures came "
            "in below forecasts for the third consecutive month."
        ),
        0.12,
        pub_date=pub_date,
        now=now,
    )
    art1b = _mock_article(
        "Fed minutes reveal growing consensus for rate hold",
        "Reuters",
        "https://example.com/1b",
        (
            "Minutes from the latest FOMC meeting show a broad majority "
            "favoring a hold, though several members warned against "
            "premature easing given persistent services inflation."
        ),
        -0.08,
        pub_date=pub_date,
        now=now,
    )
    art1c = _mock_article(
        "Markets rally as rate-cut expectations firm up",
        "Bloomberg",
        "https://example.com/1c",
        (
            "Global equities rose sharply after central bank commentary "
            "raised confidence that the tightening cycle has peaked, "
            "boosting rate-sensitive sectors including real estate and "
            "utilities."
        ),
        0.35,
        pub_date=pub_date,
        now=now,
    )
    story1 = Story(
        story_id="mock-1",
        title="Central Banks Signal End of Rate-Hike Cycle",
        articles=[art1a, art1b, art1c],
        earliest_date=pub_date,
        latest_date=now,
        story_summary=(
            "Central banks across major economies are signaling a "
            "coordinated pause on interest rate increases following "
            "encouraging inflation data. While the Federal Reserve and "
            "ECB have stopped short of committing to cuts, markets are "
            "pricing in easing by year-end. The shift marks a potential "
            "turning point after two years of aggressive tightening."
        ),
        source_additional_points={
            "Financial Times": (
                "The FT notes eurozone core inflation still running above "
                "target, making ECB caution warranted."
            ),
            "Reuters": (
                "Reuters highlights dissent within the FOMC from hawks who "
                "want further data before confirming a pivot."
            ),
            "Bloomberg": (
                "Bloomberg's analysis shows rate-sensitive ETFs saw their "
                "highest inflows in 18 months."
            ),
        },
    )

    art2a = _mock_article(
        "AI lab announces breakthrough in protein structure prediction",
        "Nature News",
        "https://example.com/2a",
        (
            "A leading AI laboratory has published results showing its new "
            "model outperforms existing tools on protein folding benchmarks "
            "by a significant margin, opening new avenues for drug discovery."
        ),
        0.55,
        pub_date=pub_date,
        now=now,
    )
    art2b = _mock_article(
        "Pharmaceutical giants rush to license new AI protein tool",
        "The Guardian",
        "https://example.com/2b",
        (
            "Several major pharmaceutical companies have entered licensing "
            "discussions after the publication of the AI breakthrough, with "
            "analysts estimating the technology could cut early-stage drug "
            "development timelines by half."
        ),
        0.28,
        pub_date=pub_date,
        now=now,
    )
    story2 = Story(
        story_id="mock-2",
        title="AI Achieves Landmark Breakthrough in Drug Discovery",
        articles=[art2a, art2b],
        earliest_date=pub_date,
        latest_date=now,
        story_summary=(
            "A new AI model has set a new standard for protein structure "
            "prediction, surpassing prior approaches on key scientific "
            "benchmarks. The advance is drawing intense interest from "
            "pharmaceutical companies who see the technology as a way to "
            "dramatically accelerate the identification of drug candidates "
            "and reduce the high failure rates in early-stage development."
        ),
        source_additional_points={
            "Nature News": (
                "The paper underwent an unusually rapid peer review, "
                "reflecting the significance of the findings."
            ),
            "The Guardian": (
                "Critics caution that lab benchmarks may not fully translate "
                "to the complexity of real-world disease targets."
            ),
        },
    )

    art3a = _mock_article(
        "Severe drought threatens grain harvests across southern Europe",
        "BBC News",
        "https://example.com/3a",
        (
            "Unusually dry conditions across the Iberian Peninsula and parts "
            "of Italy are threatening grain and olive harvests, raising "
            "concerns about food price pressures heading into winter."
        ),
        -0.42,
        pub_date=pub_date,
        now=now,
    )
    art3b = _mock_article(
        "EU activates emergency agricultural support fund",
        "Euronews",
        "https://example.com/3b",
        (
            "The European Commission has activated a €500 million emergency "
            "fund to support farmers hit by the ongoing drought, the worst "
            "in the region in over 40 years according to meteorological data."
        ),
        -0.18,
        pub_date=pub_date,
        now=now,
    )
    art3c = _mock_article(
        "Olive oil prices hit record high on supply fears",
        "Reuters",
        "https://example.com/3c",
        (
            "Futures contracts for olive oil surged to an all-time high as "
            "traders priced in supply shortfalls following damage to "
            "orchards across Spain and Portugal."
        ),
        -0.25,
        pub_date=pub_date,
        now=now,
    )
    story3 = Story(
        story_id="mock-3",
        title="European Drought Threatens Food Supply and Prices",
        articles=[art3a, art3b, art3c],
        earliest_date=pub_date,
        latest_date=now,
        story_summary=(
            "Southern Europe is experiencing its most severe drought in four "
            "decades, with major implications for grain and olive oil "
            "production. The EU has responded with emergency agricultural "
            "aid, but analysts warn that food price pressures could persist "
            "into next year if autumn rains fail to materialize."
        ),
        source_additional_points={
            "BBC News": (
                "The BBC reports that water reservoir levels in several "
                "Spanish regions have fallen below 20% capacity."
            ),
            "Euronews": (
                "The emergency fund is the third activated this decade, "
                "reflecting the increasing frequency of extreme weather "
                "events in the region."
            ),
            "Reuters": (
                "Spain accounts for over 40% of global olive oil production, "
                "making local crop failures globally significant."
            ),
        },
    )

    return [
        StoryAnalysis(
            story=story1,
            source_summaries={
                "Financial Times": _summary_items(art1a),
                "Reuters": _summary_items(art1b),
                "Bloomberg": _summary_items(art1c),
            },
            source_sentiments={
                "Financial Times": _mock_sentiment([art1a]),
                "Reuters": _mock_sentiment([art1b]),
                "Bloomberg": _mock_sentiment([art1c]),
            },
        ),
        StoryAnalysis(
            story=story2,
            source_summaries={
                "Nature News": _summary_items(art2a),
                "The Guardian": _summary_items(art2b),
            },
            source_sentiments={
                "Nature News": _mock_sentiment([art2a]),
                "The Guardian": _mock_sentiment([art2b]),
            },
        ),
        StoryAnalysis(
            story=story3,
            source_summaries={
                "BBC News": _summary_items(art3a),
                "Euronews": _summary_items(art3b),
                "Reuters": _summary_items(art3c),
            },
            source_sentiments={
                "BBC News": _mock_sentiment([art3a]),
                "Euronews": _mock_sentiment([art3b]),
                "Reuters": _mock_sentiment([art3c]),
            },
        ),
    ]
