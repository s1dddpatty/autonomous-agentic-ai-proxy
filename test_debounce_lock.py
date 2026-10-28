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

async def test_sliding_window_debounce(monkeypatch):
    chat_id = "test_user_777"
    clear_pending_messages(chat_id)
    ACTIVE_TIMERS.clear()

    invocations = []
    def mock_burst(sender, joined, lines):
        invocations.append((sender, joined, lines))

    import src.core.orchestrator
    monkeypatch.setattr(src.core.orchestrator, "handle_incoming_whatsapp_burst", mock_burst)
    # Set debounce window to 0.4 seconds of silence
    monkeypatch.setattr("src.gateways.whatsapp_webhook.DEBOUNCE_SECONDS", 0.4)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Msg 1 at t = 0
        await client.post("/webhook", json={
            "entry": [{"changes": [{"value": {"messages": [{"from": chat_id, "type": "text", "text": {"body": "Msg 1"}}]}}]}]
        })
        # Wait 0.25s (< 0.4s)
        await asyncio.sleep(0.25)
        assert len(invocations) == 0, "Timer should not have fired yet"

        # Msg 2 at t = 0.25s (resets timer for another 0.4s -> expires at 0.65s)
        await client.post("/webhook", json={
            "entry": [{"changes": [{"value": {"messages": [{"from": chat_id, "type": "text", "text": {"body": "Msg 2"}}]}}]}]
        })
        # Wait 0.25s (total t = 0.50s. Under old fixed timer, it would have cut off at 0.4s!)
        await asyncio.sleep(0.25)
        assert len(invocations) == 0, "Timer should have slid and NOT fired at 0.50s"

        # Msg 3 at t = 0.50s (resets timer for another 0.4s -> expires at 0.90s)
        await client.post("/webhook", json={
            "entry": [{"changes": [{"value": {"messages": [{"from": chat_id, "type": "text", "text": {"body": "Msg 3"}}]}}]}]
        })
        # Wait 0.25s (total t = 0.75s)
        await asyncio.sleep(0.25)
        assert len(invocations) == 0, "Timer should have slid again and NOT fired at 0.75s"

        # Now wait for the silence window to complete (0.25s more -> total t = 1.00s > 0.90s)
        await asyncio.sleep(0.30)
        assert len(invocations) == 1, f"Expected 1 unified burst, got {len(invocations)}"
        assert invocations[0][1] == "Msg 1\nMsg 2\nMsg 3"
        assert chat_id not in ACTIVE_TIMERS
        print("✅ test_sliding_window_debounce passed! Rapid messages kept resetting timer into 1 batch.")

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
    patcher = DummyMonkeyPatch()
    asyncio.run(test_debounce_lock_concurrency(patcher))
    asyncio.run(test_sliding_window_debounce(patcher))
    print("All tests completed successfully!")
