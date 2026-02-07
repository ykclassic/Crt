name: Deploy Aegis Dashboard

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 * * * *'  # Runs hourly to update stats dynamically
  workflow_dispatch:      # Allows manual trigger from GitHub UI

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pandas plotly sqlite3

      - name: Generate Dashboard
        run: python generate_dashboard.py

      - name: Deploy to GitHub Pages
        # This ensures the outcome is updated on your site
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./
          publish_branch: gh-pages
