#!/usr/bin/env python3
"""
Validate that our generated route files match the expected XML structure.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

def validate_xml_structure(xml_path: Path) -> bool:
    """Validate XML has correct structure."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Check root element
        if root.tag != 'routes':
            print(f"  [ERROR] Root element should be 'routes', got '{root.tag}'")
            return False
        
        # Check route element
        route = root.find('route')
        if route is None:
            print(f"  [ERROR] No 'route' element found")
            return False
        
        # Check required attributes
        required_attrs = ['id', 'town', 'role']
        for attr in required_attrs:
            if attr not in route.attrib:
                print(f"  [ERROR] Missing required attribute: {attr}")
                return False
        
        # Check waypoints
        waypoints = list(route.iter('waypoint'))
        if not waypoints:
            print(f"  [ERROR] No waypoints found")
            return False
        
        # Validate each waypoint
        for i, wp in enumerate(waypoints):
            required_wp_attrs = ['x', 'y', 'z', 'yaw']
            for attr in required_wp_attrs:
                if attr not in wp.attrib:
                    print(f"  [ERROR] Waypoint {i} missing attribute: {attr}")
                    return False
                
                # Try to parse as float
                try:
                    float(wp.attrib[attr])
                except ValueError:
                    print(f"  [ERROR] Waypoint {i} attribute {attr} not a valid float: {wp.attrib[attr]}")
                    return False
        
        return True
        
    except ET.ParseError as e:
        print(f"  [ERROR] XML parse error: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] Unexpected error: {e}")
        return False


def validate_manifest(manifest_path: Path) -> bool:
    """Validate actors_manifest.json structure."""
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Check top-level keys
        expected_keys = ['ego', 'npc', 'static', 'pedestrian', 'bicycle']
        for key in expected_keys:
            if key not in manifest:
                print(f"  [ERROR] Missing key in manifest: {key}")
                return False
            
            if not isinstance(manifest[key], list):
                print(f"  [ERROR] Key '{key}' should be a list")
                return False
        
        # Validate each actor entry
        for role_key, actors in manifest.items():
            for actor in actors:
                # Check required fields
                required_fields = ['file', 'route_id', 'town', 'name', 'kind']
                for field in required_fields:
                    if field not in actor:
                        print(f"  [ERROR] Actor missing field '{field}': {actor}")
                        return False
                
                # Verify file exists
                actor_file = manifest_path.parent / actor['file']
                if not actor_file.exists():
                    print(f"  [ERROR] Actor file not found: {actor_file}")
                    return False
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"  [ERROR] JSON parse error: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] Unexpected error: {e}")
        return False


def test_scenario(scenario_dir: Path) -> bool:
    """Test a single scenario directory."""
    print(f"\n{'='*80}")
    print(f"Testing: {scenario_dir.name}")
    print(f"{'='*80}")
    
    manifest_path = scenario_dir / "actors_manifest.json"
    if not manifest_path.exists():
        print(f"  [ERROR] Manifest not found")
        return False
    
    # Validate manifest
    print(f"  Validating manifest...")
    if not validate_manifest(manifest_path):
        return False
    print(f"  ✓ Manifest structure valid")
    
    # Load manifest to get list of files
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Validate all XML files
    xml_files = []
    
    # Get ego and npc vehicle files
    for xml_file in scenario_dir.glob("*.xml"):
        xml_files.append(xml_file)
    
    # Get actor files from manifest
    for role_key, actors in manifest.items():
        for actor in actors:
            xml_path = scenario_dir / actor['file']
            if xml_path not in xml_files:
                xml_files.append(xml_path)
    
    print(f"  Found {len(xml_files)} XML files to validate")
    
    for xml_file in xml_files:
        print(f"    Validating {xml_file.name}...")
        if not validate_xml_structure(xml_file):
            return False
    
    print(f"  ✓ All {len(xml_files)} XML files valid")
    
    # Summary
    ego_count = len(manifest['ego'])
    npc_count = len(manifest['npc'])
    static_count = len(manifest['static'])
    ped_count = len(manifest['pedestrian'])
    bike_count = len(manifest['bicycle'])
    
    print(f"\n  Summary:")
    print(f"    - {ego_count} ego vehicle(s)")
    print(f"    - {npc_count} NPC vehicle(s)")
    print(f"    - {static_count} static actor(s)")
    print(f"    - {ped_count} pedestrian(s)")
    print(f"    - {bike_count} bicycle(s)")
    
    return True


def main():
    batch_dir = Path(__file__).parent / "batch_routes_output"
    
    if not batch_dir.exists():
        print(f"[ERROR] Batch output directory not found: {batch_dir}")
        return 1
    
    scenarios = sorted([d for d in batch_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(scenarios)} scenarios to validate")
    
    success_count = 0
    fail_count = 0
    
    for scenario_dir in scenarios:
        try:
            if test_scenario(scenario_dir):
                success_count += 1
                print(f"  ✓ PASS")
            else:
                fail_count += 1
                print(f"  ✗ FAIL")
        except Exception as e:
            fail_count += 1
            print(f"  ✗ FAIL: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {success_count}/{len(scenarios)} scenarios passed")
    print(f"{'='*80}")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
