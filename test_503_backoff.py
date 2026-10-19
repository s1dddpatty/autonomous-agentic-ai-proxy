import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import src.core.gemini_node as gn
import src.core.orchestrator as orch

def run_tests():
    print("=== PART 1: Unit Test Simulating 2x 503 UNAVAILABLE followed by 200 OK ===")
    gn._active_key_index = 0
    call_keys = []
    sleep_calls = []

    class MockPart:
        thought = False
        text = "Haan bol na, kya hua?"

    class MockCandidate:
        content = MagicMock()
        content.parts = [MockPart()]

    class MockResponse:
        candidates = [MockCandidate()]
        text = "Haan bol na, kya hua?"

    def mock_503_factory(api_key):
        call_keys.append(api_key)
        client = MagicMock()
        if len(call_keys) <= 2:
            def raise_503(*args, **kwargs):
                raise Exception("ServerError: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary.'}}")
            client.models.generate_content.side_effect = raise_503
        else:
            client.models.generate_content.return_value = MockResponse()
        return client

    def mock_sleep(seconds):
        sleep_calls.append(seconds)

    with patch("src.core.gemini_node.genai.Client", side_effect=mock_503_factory), \
         patch("src.core.gemini_node.time.sleep", side_effect=mock_sleep):
        resp = gn.generate_with_key_rotation(contents="test 503 backoff")
        print(f"Resulting draft: '{resp}'")
        print(f"Sleep delays called: {sleep_calls}")
        print(f"Keys invoked: {[k[:6] + '...' + k[-4:] for k in call_keys]}")
        print(f"Final _active_key_index: {gn._active_key_index}")

        assert resp == "Haan bol na, kya hua?", f"Unexpected response: {resp}"
        assert sleep_calls == [2.0, 4.0], f"Expected sleeps [2.0, 4.0], got {sleep_calls}"
        assert len(call_keys) == 3, f"Expected 3 calls, got {len(call_keys)}"
        assert gn._active_key_index == 2, f"Expected active index 2, got {gn._active_key_index}"
        print("✅ Unit Test 1 PASSED: 503 high-demand spike handled with 2.0s & 4.0s backoff, rotated to Key 3, succeeded!")

    print("\n=== PART 2: Unit Test Error-Card Regeneration & Memory Protection ===")
    error_state = {
        "incoming_text": "achaa suno",
        "draft_text": "⚠️ [GEMINI ERROR: ServerError: 503 UNAVAILABLE]",
        "human_feedback": "retry",
        "messages": []
    }
    with patch("src.core.orchestrator.gemini_proxy_node") as mock_proxy, \
         patch("src.core.memory_manager.log_feedback") as mock_log:
        mock_proxy.return_value = {"draft_text": "Ha bolnaa\nSun raha hu...", "status": "pending"}
        res = orch.process_human_feedback(error_state)
        print(f"Regenerated draft from error card: {res['draft_text']}")
        assert res["draft_text"] == "Ha bolnaa\nSun raha hu..."
        mock_log.assert_not_called()
        print("✅ Unit Test 2 PASSED: Error card regenerated clean draft via gemini_proxy_node without logging to feedback_log!")

    print("\n=== PART 3: Live gemini_proxy_node Verification ('oki') ===")
    gn._active_key_index = 0
    state_input = {"incoming_text": "oki", "messages": []}
    live_res = gn.gemini_proxy_node(state_input)
    print("\nLive Generated Draft for 'oki':")
    print("-" * 50)
    print(live_res.get("draft_text", ""))
    print("-" * 50)
    print(f"Active key index after live call: Key #{gn._active_key_index + 1} (index {gn._active_key_index})")

if __name__ == "__main__":
    run_tests()
