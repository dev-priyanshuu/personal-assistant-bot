import json
import os
from app import PersonalAssistant
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_tests():
    """Run test cases and compare with expected outputs"""
    
    # Initialize assistant
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables")
        print("Please create a .env file with your Google API key:")
        print("GOOGLE_API_KEY=your_api_key_here")
        return
    
    assistant = PersonalAssistant(api_key)
    
    # Load test cases
    with open('samples/test_cases.json', 'r') as f:
        test_data = json.load(f)
    
    results = []
    passed = 0
    total = len(test_data['test_cases'])
    
    print(f"Running {total} test cases...\n")
    
    for i, test_case in enumerate(test_data['test_cases'], 1):
        print(f"Test {i}: {test_case['input'][:50]}...")
        
        try:
            # Process the request
            result = assistant.process_request(test_case['input'])
            expected = test_case['expected_output']
            
            # Check intent category
            intent_match = result['intent_category'] == expected['intent_category']
            
            # Check confidence score (within reasonable range)
            confidence_diff = abs(result['confidence_score'] - expected['confidence_score'])
            confidence_match = confidence_diff <= 0.2
            
            # Check if key entities are extracted (at least some)
            expected_entities = set(expected['key_entities'].keys())
            actual_entities = set(result['key_entities'].keys())
            entity_overlap = len(expected_entities.intersection(actual_entities))
            entity_match = entity_overlap >= len(expected_entities) * 0.5  # At least 50% overlap
            
            # Overall pass/fail
            test_passed = intent_match and confidence_match and entity_match
            
            if test_passed:
                passed += 1
                print("✅ PASSED")
            else:
                print("❌ FAILED")
                print(f"   Intent: Expected '{expected['intent_category']}', Got '{result['intent_category']}'")
                print(f"   Confidence: Expected {expected['confidence_score']}, Got {result['confidence_score']}")
                print(f"   Entities overlap: {entity_overlap}/{len(expected_entities)}")
            
            results.append({
                'test_case': i,
                'input': test_case['input'],
                'passed': test_passed,
                'expected': expected,
                'actual': result
            })
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append({
                'test_case': i,
                'input': test_case['input'],
                'passed': False,
                'error': str(e)
            })
        
        print()  # Empty line for readability
    
    # Print summary
    print("="*50)
    print(f"Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    print("="*50)
    
    # Save detailed results
    with open('test_results.json', 'w') as f:
        json.dump({
            'summary': {
                'total_tests': total,
                'passed': passed,
                'failed': total - passed,
                'pass_rate': passed/total*100
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"Detailed results saved to test_results.json")
    
    return results

def run_single_test(input_text: str):
    """Run a single test case"""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables")
        return
    
    assistant = PersonalAssistant(api_key)
    result = assistant.process_request(input_text)
    
    print(f"Input: {input_text}")
    print(f"Output:")
    print(json.dumps(result, indent=2))
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the Personal Assistant Bot')
    parser.add_argument('--single', type=str, help='Run a single test with given input')
    parser.add_argument('--all', action='store_true', help='Run all test cases')
    
    args = parser.parse_args()
    
    if args.single:
        run_single_test(args.single)
    elif args.all:
        run_tests()
    else:
        print("Usage:")
        print("  python test_runner.py --all              # Run all test cases")
        print("  python test_runner.py --single 'input'   # Run single test")
        print("\nExample:")
        print("  python test_runner.py --single 'Need a table for 2 tonight'")