import os
import sys
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import src.core.gemini_node as gn

def test_all():
    print(f"=== PART 1: Loaded Keys ({len(gn.GEMINI_API_KEYS)} total) ===")
    for idx, key in enumerate(gn.GEMINI_API_KEYS):
        if len(key) > 8:
            masked = f"{key[:6]}...{key[-4:]}"
        else:
            masked = "***"
        print(f"Key #{idx + 1}: {masked}")
    assert len(gn.GEMINI_API_KEYS) == 5, f"Expected 5 keys, got {len(gn.GEMINI_API_KEYS)}"

    print("\n=== PART 2: Unit Test Simulating 429 Quota Exhaustion on Key #1 ===")
    gn._active_key_index = 0
    call_history = []

    class MockPart:
        thought = False
        text = "Unit test response from Key #2"

    class MockCandidate:
        content = MagicMock()
        content.parts = [MockPart()]

    class MockResponse:
        candidates = [MockCandidate()]
        text = "Unit test response from Key #2"

    def mock_client_factory(api_key):
        call_history.append(api_key)
        client = MagicMock()
        if api_key == gn.GEMINI_API_KEYS[0]:
            def raise_quota(*args, **kwargs):
                raise Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric 'Generate Content API requests'")
            client.models.generate_content.side_effect = raise_quota
        else:
            client.models.generate_content.return_value = MockResponse()
        return client

    with patch("src.core.gemini_node.genai.Client", side_effect=mock_client_factory):
        resp_text = gn.generate_with_key_rotation(contents="test prompt")
        print(f"Mock rotation response: '{resp_text}'")
        print(f"Called keys in order: {[k[:6] + '...' + k[-4:] for k in call_history]}")
        print(f"Resulting _active_key_index: {gn._active_key_index}")
        assert gn._active_key_index == 1, f"Expected _active_key_index to be 1, got {gn._active_key_index}"
        assert resp_text == "Unit test response from Key #2"
        print("✅ Unit test PASSED: Key #1 hit 429, rotated cleanly to Key #2, pointer advanced to 1.")

    print("\n=== PART 2b: Unit Test Simulating All 5 Keys Exhausted ===")
    gn._active_key_index = 0
    def mock_all_fail_factory(api_key):
        client = MagicMock()
        def raise_quota(*args, **kwargs):
            raise Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for project")
        client.models.generate_content.side_effect = raise_quota
        return client

    with patch("src.core.gemini_node.genai.Client", side_effect=mock_all_fail_factory):
        fallback_resp = gn.generate_with_key_rotation(contents="test prompt")
        print(f"Fallback response:\n{fallback_resp}")
        assert gn.is_quota_fallback_text(fallback_resp), "Expected response to be recognized as quota fallback"
        print("✅ Unit test PASSED: All 5 keys failed with 429, returned in-character quota fallback.")

    print("\n=== PART 3: Live gemini_proxy_node Verification ===")
    gn._active_key_index = 0
    print(f"Reset _active_key_index to 0 to test real-world rotation on model '{gn.MODEL_NAME}'...")
    state_input = {"incoming_text": "achaa suno", "messages": []}
    result = gn.gemini_proxy_node(state_input)
    print("\nLive Generated Draft:")
    print("-" * 50)
    print(result.get("draft_text", ""))
    print("-" * 50)
    print(f"Active key index after live call: Key #{gn._active_key_index + 1} (index {gn._active_key_index})")

if __name__ == "__main__":
    test_all()
