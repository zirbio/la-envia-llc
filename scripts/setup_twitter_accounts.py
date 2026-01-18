#!/usr/bin/env python3
"""
Setup script for configuring Twitter accounts with twscrape.

This script reads Twitter credentials from environment variables and
configures them in twscrape's account pool for scraping Twitter data.
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from twscrape import AccountsPool


async def setup_twitter_accounts():
    """Configure Twitter accounts in twscrape pool."""
    # Load environment variables
    load_dotenv()

    # Get credentials from environment
    username = os.getenv("TWITTER_USERNAME")
    password = os.getenv("TWITTER_PASSWORD")
    email = os.getenv("TWITTER_EMAIL")
    email_password = os.getenv("TWITTER_EMAIL_PASSWORD")

    # Validate credentials
    if not all([username, password, email, email_password]):
        missing = []
        if not username:
            missing.append("TWITTER_USERNAME")
        if not password:
            missing.append("TWITTER_PASSWORD")
        if not email:
            missing.append("TWITTER_EMAIL")
        if not email_password:
            missing.append("TWITTER_EMAIL_PASSWORD")

        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please configure them in your .env file"
        )

    # Initialize account pool
    # twscrape stores accounts in accounts.db by default
    pool = AccountsPool()

    print(f"Adding Twitter account: @{username}")

    # Add account to pool
    await pool.add_account(
        username=username,
        password=password,
        email=email,
        email_password=email_password,
    )

    print("Logging into Twitter account...")

    # Login to all accounts in the pool
    await pool.login_all()

    print("✓ Twitter account configured successfully!")
    print(f"✓ Account @{username} is ready to use")

    # Display account stats
    accounts = await pool.accounts_info()
    print(f"\nTotal accounts in pool: {len(accounts)}")
    for acc in accounts:
        status = "Active" if acc.get("active") else "Inactive"
        print(f"  - @{acc.get('username')}: {status}")


async def verify_setup():
    """Verify that the Twitter account is properly configured."""
    pool = AccountsPool()
    accounts = await pool.accounts_info()

    if not accounts:
        print("❌ No accounts found in pool")
        return False

    print("\n--- Account Verification ---")
    for acc in accounts:
        print(f"Username: @{acc.get('username')}")
        print(f"Email: {acc.get('email')}")
        print(f"Active: {acc.get('active')}")
        print(f"Last used: {acc.get('last_used', 'Never')}")
        print(f"Error msg: {acc.get('error_msg', 'None')}")
        print("---")

    return True


async def main():
    """Main entry point."""
    try:
        await setup_twitter_accounts()
        await verify_setup()
    except Exception as e:
        print(f"❌ Error setting up Twitter accounts: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
