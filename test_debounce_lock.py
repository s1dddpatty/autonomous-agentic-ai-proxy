import asyncio
import time
from httpx import AsyncClient, ASGITransport
from src.core.database import add_pending_message, pop_pending_messages, clear_pending_messages
from src.gateways.whatsapp_webhook import app, ACTIVE_TIMERS, debounce_and_dispatch, DEBOUNCE_SECONDS

async def test_sqlite_pending_buffer():
    chat_id = "test_user_999"
    clear_pending_messages(chat_id)
    assert pop_pending_messages(chat_id) == []

    # Add 3 messages
    add_pending_message(chat_id, "Hey there")
    add_pending_message(chat_id, "Are you awake?")
    add_pending_message(chat_id, "Call me when you see this")

    # Pop should return all in order and clear them
    popped = pop_pending_messages(chat_id)
    assert popped == ["Hey there", "Are you awake?", "Call me when you see this"]
    assert pop_pending_messages(chat_id) == []
    print("✅ test_sqlite_pending_buffer passed")

async def test_debounce_lock_concurrency(monkeypatch):
    chat_id = "test_user_888"
    clear_pending_messages(chat_id)
    ACTIVE_TIMERS.clear()

    invocations = []
    def mock_burst(sender, joined, lines):
        invocations.append((sender, joined, lines))

    import src.core.orchestrator
    monkeypatch.setattr(src.core.orchestrator, "handle_incoming_whatsapp_burst", mock_burst)
    monkeypatch.setattr("src.gateways.whatsapp_webhook.DEBOUNCE_SECONDS", 0.3)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First burst: 3 messages
        for text in ["Burst 1 - Msg 1", "Burst 1 - Msg 2", "Burst 1 - Msg 3"]:
            payload = {
                "entry": [{"changes": [{"value": {"messages": [{"from": chat_id, "type": "text", "text": {"body": text}}]}}]}]
            }
            res = await client.post("/webhook", json=payload)
            assert res.status_code == 200

        # Verify active timer is running
        assert chat_id in ACTIVE_TIMERS

        # Wait for the first debounce timer to finish (0.3s sleep + margin)
        await asyncio.sleep(0.4)

        # Verify timer is strictly removed from ACTIVE_TIMERS
        assert chat_id not in ACTIVE_TIMERS
        assert len(invocations) == 1
        assert invocations[0][1] == "Burst 1 - Msg 1\nBurst 1 - Msg 2\nBurst 1 - Msg 3"

        # Second burst: 2 messages arriving after first burst completed
        for text in ["Burst 2 - Msg 1", "Burst 2 - Msg 2"]:
            payload = {
                "entry": [{"changes": [{"value": {"messages": [{"from": chat_id, "type": "text", "text": {"body": text}}]}}]}]
            }
            res = await client.post("/webhook", json=payload)
            assert res.status_code == 200

        assert chat_id in ACTIVE_TIMERS
        await asyncio.sleep(0.4)
        assert chat_id not in ACTIVE_TIMERS
        assert len(invocations) == 2
        assert invocations[1][1] == "Burst 2 - Msg 1\nBurst 2 - Msg 2"

        print("✅ test_debounce_lock_concurrency & second burst passed")

if __name__ == "__main__":
    asyncio.run(test_sqlite_pending_buffer())
    class DummyMonkeyPatch:
        def setattr(self, target, name_or_val, val=None):
            if val is None:
                parts = target.rsplit(".", 1)
                import importlib
                mod = importlib.import_module(parts[0])
                setattr(mod, parts[1], name_or_val)
            else:
                setattr(target, name_or_val, val)
    asyncio.run(test_debounce_lock_concurrency(DummyMonkeyPatch()))
    print("All tests completed successfully!")
