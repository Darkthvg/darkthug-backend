#!/usr/bin/env python3
"""
DARKTHUG AI License Generator
Military-grade license key generation with HMAC-SHA256 validation
"""

import hmac
import hashlib
import secrets
import argparse
import os
from datetime import datetime, timedelta

# Secret key for HMAC (use same as backend SECRET_KEY)
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-minimum-32-characters-long")

TIER_CODES = {
    "demo": "D",
    "trial": "T",
    "premium": "P",
    "master": "M"
}

def generate_license(tier: str = "demo", duration_days: int = None) -> dict:
    """
    Generate a military-grade license key
    
    Format: DTHUG-{TIER}{RANDOM}-{RANDOM}-{CHECKSUM}-{TIMESTAMP}
    Example: DTHUG-M7K4J-8H3F2-L9P6N-Q2W5X
    """
    
    if tier not in TIER_CODES:
        raise ValueError(f"Invalid tier. Must be one of: {list(TIER_CODES.keys())}")
    
    tier_code = TIER_CODES[tier]
    
    # Generate cryptographically secure random parts
    part1 = tier_code + ''.join(secrets.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(4))
    part2 = ''.join(secrets.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(5))
    
    # Generate timestamp component (encoded)
    timestamp = int(datetime.utcnow().timestamp())
    ts_encoded = format(timestamp % 100000, '05d')  # Last 5 digits
    
    # Create checksum using HMAC-SHA256
    data_to_sign = f"{part1}{part2}{ts_encoded}"
    checksum = hmac.new(
        SECRET_KEY.encode(),
        data_to_sign.encode(),
        hashlib.sha256
    ).hexdigest()[:5].upper()
    
    # Construct license key
    license_key = f"DTHUG-{part1}-{part2}-{checksum}-{ts_encoded}"
    
    # Calculate expiration if applicable
    expires_at = None
    if duration_days:
        expires_at = datetime.utcnow() + timedelta(days=duration_days)
    
    # Tier quotas
    quotas = {
        "demo": 10,
        "trial": 100,
        "premium": 1000,
        "master": None  # Unlimited
    }
    
    return {
        "license_key": license_key,
        "tier": tier,
        "daily_quota": quotas[tier],
        "expires_at": expires_at.isoformat() if expires_at else None,
        "created_at": datetime.utcnow().isoformat(),
        "features": {
            "unrestricted": tier in ["premium", "master"],
            "rag_access": tier in ["trial", "premium", "master"],
            "code_execution": tier in ["premium", "master"],
            "web_search": tier in ["premium", "master"]
        }
    }

def validate_license(license_key: str) -> bool:
    """Validate a license key's format and checksum"""
    
    if not license_key.startswith("DTHUG-"):
        return False
    
    parts = license_key.split("-")
    if len(parts) != 5:
        return False
    
    prefix, part1, part2, checksum, timestamp = parts
    
    # Verify checksum
    data_to_sign = f"{part1}{part2}{timestamp}"
    expected_checksum = hmac.new(
        SECRET_KEY.encode(),
        data_to_sign.encode(),
        hashlib.sha256
    ).hexdigest()[:5].upper()
    
    return checksum == expected_checksum

def get_tier_from_license(license_key: str) -> str:
    """Extract tier from license key"""
    if not license_key.startswith("DTHUG-"):
        return None
    
    tier_code = license_key.split("-")[1][0]
    
    for tier, code in TIER_CODES.items():
        if code == tier_code:
            return tier
    
    return None

def batch_generate(tier: str, count: int, duration_days: int = None) -> list:
    """Generate multiple licenses"""
    licenses = []
    
    for i in range(count):
        license_data = generate_license(tier, duration_days)
        licenses.append(license_data)
        print(f"Generated {i+1}/{count}: {license_data['license_key']}")
    
    return licenses

def main():
    parser = argparse.ArgumentParser(description="DARKTHUG AI License Generator")
    
    parser.add_argument(
        "--tier",
        type=str,
        choices=["demo", "trial", "premium", "master"],
        default="demo",
        help="License tier (default: demo)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        help="License duration in days (omit for unlimited)"
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of licenses to generate (default: 1)"
    )
    
    parser.add_argument(
        "--validate",
        type=str,
        help="Validate an existing license key"
    )
    
    args = parser.parse_args()
    
    if args.validate:
        # Validate mode
        is_valid = validate_license(args.validate)
        tier = get_tier_from_license(args.validate)
        
        print("\n" + "="*60)
        print("LICENSE VALIDATION")
        print("="*60)
        print(f"License Key: {args.validate}")
        print(f"Valid: {'✓ YES' if is_valid else '✗ NO'}")
        if tier:
            print(f"Tier: {tier.upper()}")
        print("="*60 + "\n")
    else:
        # Generation mode
        print("\n" + "="*60)
        print("DARKTHUG AI LICENSE GENERATOR")
        print("="*60)
        print(f"Generating {args.count} {args.tier.upper()} license(s)...")
        print()
        
        if args.count == 1:
            license_data = generate_license(args.tier, args.duration)
            
            print("\n" + "="*60)
            print("LICENSE GENERATED SUCCESSFULLY")
            print("="*60)
            print(f"\n🔑 License Key: {license_data['license_key']}")
            print(f"\n📊 Details:")
            print(f"   Tier: {license_data['tier'].upper()}")
            print(f"   Daily Quota: {license_data['daily_quota'] or 'Unlimited'}")
            print(f"   Expires: {license_data['expires_at'] or 'Never'}")
            print(f"\n✨ Features:")
            for feature, enabled in license_data['features'].items():
                status = "✓" if enabled else "✗"
                print(f"   {status} {feature.replace('_', ' ').title()}")
            print("\n" + "="*60)
            print("\n⚠️  IMPORTANT: Save this license key securely!")
            print("   It cannot be recovered if lost.\n")
        else:
            licenses = batch_generate(args.tier, args.count, args.duration)
            
            print("\n" + "="*60)
            print(f"BATCH GENERATION COMPLETE: {len(licenses)} licenses")
            print("="*60 + "\n")
            
            # Save to file
            filename = f"licenses_{args.tier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                for lic in licenses:
                    f.write(f"{lic['license_key']}\n")
            
            print(f"✅ Licenses saved to: {filename}\n")

if __name__ == "__main__":
    main()