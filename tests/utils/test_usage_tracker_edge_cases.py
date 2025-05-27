import pytest
from dspy.utils.usage_tracker import UsageTracker

def test_merge_usage_entries_with_type_mismatch():
    """Test merging usage entries with type mismatches between dictionaries and non-dictionaries."""
    tracker = UsageTracker()
    
    # Create a usage entry with a dictionary value
    usage_entry1 = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "nested_field": {
            "subfield1": 10,
            "subfield2": 20
        }
    }
    
    # Create a usage entry with a non-dictionary value for the same field
    usage_entry2 = {
        "prompt_tokens": 200,
        "completion_tokens": 100,
        "total_tokens": 300,
        "nested_field": 30  # This is an int, not a dict like in usage_entry1
    }
    
    # Test merging in both directions
    # First, merge entry1 into entry2
    result1 = tracker._merge_usage_entries(usage_entry1, usage_entry2)
    # The result should keep the dictionary from usage_entry1
    assert isinstance(result1["nested_field"], dict)
    assert result1["nested_field"]["subfield1"] == 10
    assert result1["nested_field"]["subfield2"] == 20
    
    # Now, merge entry2 into entry1
    result2 = tracker._merge_usage_entries(usage_entry2, usage_entry1)
    # The result should keep the dictionary from usage_entry1
    assert isinstance(result2["nested_field"], dict)
    assert result2["nested_field"]["subfield1"] == 10
    assert result2["nested_field"]["subfield2"] == 20
    
    # Test with a more complex scenario that mimics the Bedrock model response
    bedrock_entry1 = {
        "input_tokens": 100,
        "output_tokens": 50,
        "model": {
            "name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "input_tokens": 100,
            "output_tokens": 50
        }
    }
    
    bedrock_entry2 = {
        "input_tokens": 200,
        "output_tokens": 100,
        "model": "anthropic.claude-3-sonnet-20240229-v1:0"  # String instead of dict
    }
    
    # Merge bedrock entries
    bedrock_result = tracker._merge_usage_entries(bedrock_entry1, bedrock_entry2)
    # The result should keep the dictionary for "model"
    assert isinstance(bedrock_result["model"], dict)
    assert bedrock_result["model"]["name"] == "anthropic.claude-3-sonnet-20240229-v1:0"
    assert bedrock_result["model"]["input_tokens"] == 100
    assert bedrock_result["model"]["output_tokens"] == 50
    
    # Test the full usage tracking flow
    tracker.add_usage("bedrock/claude-3", bedrock_entry1)
    tracker.add_usage("bedrock/claude-3", bedrock_entry2)
    
    total_usage = tracker.get_total_tokens()
    # Verify that we can get the total tokens without errors
    assert "bedrock/claude-3" in total_usage
    assert total_usage["bedrock/claude-3"]["input_tokens"] == 300  # 100 + 200
    assert total_usage["bedrock/claude-3"]["output_tokens"] == 150  # 50 + 100
    assert isinstance(total_usage["bedrock/claude-3"]["model"], dict)