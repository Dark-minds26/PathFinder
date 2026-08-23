from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_profile_chat_returns_not_yet_implemented():
    response = client.post(
        "/profile/chat", json={"user_id": "u1", "message": "hi"}
    )
    assert response.status_code == 501


def test_explain_returns_not_yet_implemented():
    response = client.get("/explain/course_1/u1")
    assert response.status_code == 501
