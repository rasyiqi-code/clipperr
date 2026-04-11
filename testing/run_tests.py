import unittest
import sys
import os

def run():
    print("="*60)
    print("  clipperr UNIT TEST SUITE")
    print("="*60)
    
    # Add project directories to path
    # base_dir is now the project root (one level up from this file)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(base_dir, 'app'))
    sys.path.append(base_dir) # For clipperr_core
    
    # Discover and run tests in the local 'tests' directory
    loader = unittest.TestLoader()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    start_dir = os.path.join(current_dir, 'tests')
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
