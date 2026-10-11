"""System One API と互換のサーバー。リクエスト例は TypeSafe の API リファレンスのもの。"""

import asyncio
import threading

import httpx2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    Score,
    TypeSafeUnprocessableEntityError,
)

from local_decision_model.decide import Decider
from local_decision_model.server import ModelInfo, create_app

STATE = "Help! My payouts have been failing for 3 days."
MODEL = ModelInfo("local-decision-model-test", "テスト用", "2026-10-11")
DEPARTMENT = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts",
    },
}


class FakeScorer:
    """仮説文に "billing" か "urgency" があれば高い判断ロジットを返す。受け取った入力を覚える。"""

    def __init__(self):
        self.pairs = []

    def score(self, pairs):
        self.pairs += pairs
        return np.array([3.0 if "billing" in h or "urgency" in h else -1.0 for _, h in pairs])


def make_app(scorer=None, load=None):
    app = create_app(load or (lambda: Decider(scorer or FakeScorer())), MODEL)
    return app


@pytest.fixture
def scorer():
    return FakeScorer()


@pytest.fixture
def client(scorer):
    app = make_app(scorer)
    assert app.state.loaded.wait(5)
    return TestClient(app)


def post(client, questions, state=STATE, model="jev-latest"):
    body = {"state": state, "model": model, "questions": questions}
    return client.post("/v1/systemone", json=body)


def test_noul(client):
    question = {"type": "noul", "instructions": "Does this convey urgency?"}
    response = post(client, {"is_urgent": question})
    assert response.status_code == 200
    assert response.json() == {
        "model": MODEL.name,
        "answers": {"is_urgent": {"type": "noul", "noul": pytest.approx(1 / (1 + np.exp(-3.0)))}},
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_noul_criteria_true_is_the_hypothesis(client, scorer):
    question = {
        "type": "noul",
        "instructions": "Does this convey urgency?",
        "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
    }
    assert post(client, {"is_urgent": question}).status_code == 200
    assert scorer.pairs == [(STATE, "Explicitly time-sensitive")]


def test_choice(client):
    answer = post(client, {"department": DEPARTMENT}).json()["answers"]["department"]
    assert answer["type"] == "choice"
    assert answer["choice"] == "billing"
    assert list(answer["probabilities"]) == ["billing", "technical", "sales"]
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0)
    p_max = answer["probabilities"]["billing"]
    assert answer["confidence"] == pytest.approx((3 * p_max - 1) / 2)


def test_structured_state_becomes_json(client, scorer):
    state = {"subject": "二重請求", "messages": ["a", "b"]}
    question = {"type": "noul", "instructions": "Is this about billing?"}
    assert post(client, {"q": question}, state=state).status_code == 200
    assert scorer.pairs[0][0] == '{"subject": "二重請求", "messages": ["a", "b"]}'


def test_models_and_names(client):
    names = [m["name"] for m in client.get("/v1/models").json()["models"]]
    assert names == [MODEL.name, "jev-latest"]
    question = {"q": {"type": "noul", "instructions": "Is this about billing?"}}
    for name in (MODEL.name, "local-decision-model", "jev-1.13.0"):
        assert post(client, question, model=name).json()["model"] == MODEL.name
    response = post(client, question, model="gpt-4")
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "model"]


@pytest.mark.parametrize(
    ("body", "loc"),
    [
        ({"model": "jev-latest", "questions": {"q": {"type": "noul"}}}, ["body", "state"]),
        ({"state": STATE, "model": "jev-latest", "questions": {}}, ["body", "questions"]),
        (
            {"state": STATE, "model": "jev-latest", "questions": {"q": {"type": "choice"}}},
            ["body", "questions", "q", "choice", "criteria"],
        ),
        (
            {"state": STATE, "model": "jev-latest", "questions": {"q": {"type": "noul"}}},
            ["body", "questions", "q", "noul"],
        ),
        (
            {
                "state": STATE,
                "model": "jev-latest",
                "questions": {"q": {"type": "choice", "criteria": {"only": None}}},
            },
            ["body", "questions", "q", "choice"],
        ),
        (
            {
                "state": STATE,
                "model": "jev-latest",
                "questions": {"q": {"type": "score", "criteria": ["Calm", "Angry"]}},
            },
            ["body", "questions", "q", "score"],
        ),
        (
            {
                "state": STATE,
                "model": "jev-latest",
                "questions": {"q": {"type": "noul", "instructions": {"question": "?"}}},
            },
            ["body", "questions", "q", "noul", "instructions"],
        ),
    ],
)
def test_validation_errors(client, body, loc):
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == loc
    assert error["msg"] and error["type"]


def test_loading_then_ready():
    release = threading.Event()

    def load():
        release.wait(5)
        return Decider(FakeScorer())

    app = make_app(load=load)
    client = TestClient(app)
    question = {"q": {"type": "noul", "instructions": "Is this about billing?"}}
    assert client.get("/health").status_code == 503
    assert post(client, question).status_code == 503
    # 検証エラーは、読み込み中でも返す
    assert client.post("/v1/systemone", json={}).status_code == 422

    release.set()
    assert app.state.loaded.wait(5)
    assert client.get("/health").json() == {"status": "ok", "model": MODEL.name}
    assert post(client, question).status_code == 200


def test_load_failure_is_reported():
    def load():
        raise FileNotFoundError("model.xml がありません")

    app = make_app(load=load)
    assert app.state.loaded.wait(5)
    response = TestClient(app).get("/health")
    assert response.status_code == 503
    assert "model.xml" in response.json()["detail"]


def test_requests_run_one_at_a_time():
    class SlowScorer(FakeScorer):
        running = 0
        overlapped = False

        def score(self, pairs):
            SlowScorer.running += 1
            SlowScorer.overlapped |= SlowScorer.running > 1
            threading.Event().wait(0.05)
            SlowScorer.running -= 1
            return super().score(pairs)

    app = make_app(SlowScorer())
    assert app.state.loaded.wait(5)
    client = TestClient(app)
    question = {"q": {"type": "noul", "instructions": "Is this about billing?"}}
    threads = [threading.Thread(target=post, args=(client, question)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not SlowScorer.overlapped


def sdk(app, call):
    """公式の Python SDK で、base_url をこのサーバーに向けて呼ぶ (ソケットは使わない)。"""

    async def run():
        async with AsyncTypeSafeClient(
            api_key="unused",
            base_url="http://local-decision-model",
            transport=httpx2.ASGITransport(app=app),
        ) as client:
            return await call(client)

    return asyncio.run(run())


def test_official_sdk():
    app = make_app()
    assert app.state.loaded.wait(5)

    async def ask(client):
        return await client.system_one(
            state={"document": "I was charged twice. Please fix this ASAP."},
            questions={
                "billing": Noul(instructions="Is this ticket about billing?"),
                "tone": Choice(
                    instructions="What is the customer's tone?",
                    criteria={"calm": None, "frustrated": None, "angry": None},
                ),
            },
        )

    response = sdk(app, ask)
    assert response.model == MODEL.name
    assert response.nouls["billing"].noul > 0.9
    assert response.choices["tone"].choice in {"calm", "frustrated", "angry"}
    assert 0.0 <= response.choices["tone"].confidence <= 1.0
    assert response.usage.output_tokens == 0

    async def list_models(client):
        return await client.models.list()

    assert [m.name for m in sdk(app, list_models).models] == [MODEL.name, "jev-latest"]


def test_official_sdk_gets_422_for_score():
    app = make_app()
    assert app.state.loaded.wait(5)

    async def ask(client):
        return await client.system_one(
            state="I was charged twice.",
            questions={"urgency": Score(instructions="How urgent?", criteria=["wait", "today"])},
        )

    with pytest.raises(TypeSafeUnprocessableEntityError, match="urgency"):
        sdk(app, ask)
