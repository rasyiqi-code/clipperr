import unittest
import sys
import os

def run():
    print("="*60)
    print("  clipperr UNIT TEST SUITE")
    print("="*60)
    
    # Add project directories to path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(base_dir, 'app'))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(base_dir, 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("  🎉 ALL TESTS PASSED!")
    else:
        print(f"  ❌ FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("="*60)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run()
    sys.exit(0 if success else 1)
