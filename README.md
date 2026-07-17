---
title: Brawl Stars Drafter
emoji: 🎮
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---
# Brawl Stars ML Draft Predictor

An advanced draft predictor for Brawl Stars ranked matches. The project uses a machine learning classifier (XGBoost) combined with a Minimax game-theoretic search to recommend the optimal brawler selections dynamically during a draft.

## Project Structure

*   `scraper.py`: A snowball crawler that connects to the Brawl Stars API to discover players and record ranked match logs, writing them to `india_ranked_drafts.csv`.
*   `train_model.py`: Processes and cleans raw match data, one-hot encodes map selections, converts drafts into team representation vectors, trains an XGBoost model with early stopping, and exports the model state to `draft_model.pkl`.
*   `predictor.py`: Basic command-line predictor simulating greedy picks for any given map, allies, and enemies.
*   `minimax_predictor.py`: Implements a vectorized minimax algorithm to identify optimal picks that maximize win probability against counter-picks.
*   `smart_predictor.py`: A unified controller that dynamically switches between Greedy and Minimax logic depending on whether the enemy team is fully locked in.
*   `server.py`: A Flask backend serving predictions and API proxy endpoints.
*   `New folder/`: Interactive web UI managing the draft process, showing live ML suggestions.

---

## Installation & Setup

### 1. Install Dependencies
Make sure you have Python installed, then install the package dependencies:
```bash
pip install -r pyproject.toml
```
*Note: If you use the `uv` tool, you can simply run:*
```bash
uv pip install -e .
```

### 2. Configure Environment Variables
1. Copy the environment template file:
   ```bash
   cp .env.example .env
   ```
2. Open the new `.env` file and replace `"your_actual_api_key_here"` with your Supercell developer token (which you can generate from the [Brawl Stars Developer Portal](https://developer.brawlstars.com/)).

---

## How to Run

### Step 1: Scrape Ranked Match Data
If you don't already have an up-to-date dataset, gather match results from high-ranking players by running:
```bash
python scraper.py
```
This runs a snowball search crawl through live battle logs and saves outputs directly to `india_ranked_drafts.csv`.

### Step 2: Train the ML Model
To clean the data and train the XGBoost model:
```bash
python train_model.py
```
Upon successful completion, this outputs evaluation metrics (accuracy) and saves the model data to `draft_model.pkl`.

### Step 3: Launch the Backend Server
Start the Flask backend server on port 5000:
```bash
python server.py
```

### Step 4: Open the Draft UI
Once the server is running, open the file `New folder/index.html` in any web browser. Enter the target map name and press Enter to initialize the draft board and begin drafting with real-time recommendations.
