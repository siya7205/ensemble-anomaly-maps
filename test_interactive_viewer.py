#!/usr/bin/env python3
"""
Test script for interactive molecular visualization features.
Validates API endpoints and data loading.
"""

import sys
import requests
import json
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

def test_api(base_url="http://localhost:5051"):
    """Test all API endpoints."""
    
    print("=" * 60)
    print("Testing Interactive Molecular Visualization APIs")
    print("=" * 60)
    
    tests = [
        ("Trajectory Metadata", "/api/trajectory/meta"),
        ("Residue Map", "/api/trajectory/residue_map"),
        ("All Residues", "/api/residues"),
        ("Hotspot Data", "/api/hotspots"),
        ("RMSF Data", "/api/rmsf"),
        ("Frame 0 Coordinates", "/api/trajectory/frame/0"),
        ("Residue 10 Details", "/api/residue/10"),
        ("Atom 100 Info", "/api/atom/100"),
    ]
    
    results = []
    
    for name, endpoint in tests:
        print(f"\n📡 Testing: {name}")
        print(f"   Endpoint: {endpoint}")
        
        try:
            resp = requests.get(f"{base_url}{endpoint}", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                
                # Print summary based on endpoint
                if "meta" in endpoint:
                    print(f"   ✅ Success: {data.get('n_frames')} frames, {data.get('n_atoms')} atoms, {data.get('n_residues')} residues")
                elif "residue_map" in endpoint:
                    resnums = data.get('resnums', [])
                    print(f"   ✅ Success: Got residue map for {len(resnums)} atoms")
                elif "residues" in endpoint and "api/residues" in endpoint:
                    residues = data.get('residues', [])
                    print(f"   ✅ Success: {len(residues)} residues")
                elif "hotspots" in endpoint:
                    hotspots = data.get('hotspots', {})
                    print(f"   ✅ Success: {len(hotspots)} hotspot entries")
                    if hotspots:
                        top_res = max(hotspots.items(), key=lambda x: x[1].get('score', 0))
                        print(f"   🔥 Top hotspot: Residue {top_res[0]} (score: {top_res[1].get('score', 0):.4f})")
                elif "rmsf" in endpoint:
                    rmsf = data.get('rmsf', {})
                    print(f"   ✅ Success: {len(rmsf)} RMSF entries")
                elif "frame" in endpoint:
                    xyz = data.get('xyz', [])
                    print(f"   ✅ Success: {len(xyz)} atom coordinates")
                elif "residue" in endpoint and endpoint.endswith(('10', '36', '40')):
                    print(f"   ✅ Success: {data.get('resname')} {data.get('resid')}, chain {data.get('chain')}")
                    if data.get('hotspot'):
                        print(f"   🔥 Hotspot score: {data['hotspot'].get('score', 'N/A')}")
                elif "atom" in endpoint:
                    print(f"   ✅ Success: {data.get('name')} ({data.get('element')}), residue {data.get('resid')}")
                else:
                    print(f"   ✅ Success: {len(json.dumps(data))} bytes")
                
                results.append((name, True, None))
            else:
                error_msg = f"HTTP {resp.status_code}"
                print(f"   ❌ Failed: {error_msg}")
                results.append((name, False, error_msg))
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"\n✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        print("\nFailed tests:")
        for name, success, error in results:
            if not success:
                print(f"  - {name}: {error}")
    
    return passed == total


def test_data_loader():
    """Test the data loader directly."""
    
    print("\n" + "=" * 60)
    print("Testing Data Loader Module")
    print("=" * 60)
    
    try:
        from app.data_loader import get_loader
        
        loader = get_loader()
        
        print("\n📊 Loading trajectory metadata...")
        meta = loader.get_trajectory_meta()
        print(f"   Frames: {meta['n_frames']}")
        print(f"   Atoms: {meta['n_atoms']}")
        print(f"   Residues: {meta['n_residues']}")
        print(f"   Chains: {meta['n_chains']}")
        
        print("\n🔥 Loading hotspot data...")
        hotspots = loader.load_hotspots()
        print(f"   Loaded {len(hotspots)} hotspot scores")
        
        if hotspots:
            top_5 = sorted(hotspots.items(), key=lambda x: x[1].get('score', 0), reverse=True)[:5]
            print("\n   Top 5 Hotspots:")
            for resid, data in top_5:
                print(f"     Residue {resid}: score={data.get('score', 0):.4f}, rank={data.get('rank', 'N/A')}")
        
        print("\n📈 Loading RMSF data...")
        rmsf = loader.load_rmsf()
        print(f"   Loaded {len(rmsf)} RMSF entries")
        
        print("\n✅ Data loader tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Data loader test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test interactive viewer features")
    parser.add_argument("--url", default="http://localhost:5051", help="Base URL for API tests")
    parser.add_argument("--skip-api", action="store_true", help="Skip API tests (requires running server)")
    args = parser.parse_args()
    
    # Always test data loader
    loader_ok = test_data_loader()
    
    # Test APIs if requested
    if not args.skip_api:
        print("\n" + "=" * 60)
        print("⚠️  Note: API tests require Flask server running!")
        print(f"   Start with: python app/app.py")
        print("=" * 60)
        
        try:
            api_ok = test_api(args.url)
        except requests.exceptions.ConnectionError:
            print("\n❌ Could not connect to Flask server")
            print(f"   Make sure it's running on {args.url}")
            api_ok = False
    else:
        api_ok = True
    
    # Exit with appropriate code
    sys.exit(0 if (loader_ok and api_ok) else 1)
