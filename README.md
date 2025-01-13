# NCAA March Madness Predictive Model

## Overview

This Python-based model analyzes NCAA men’s basketball tournament data sourced from KenPom.com to generate predictive insights for March Madness tournament outcomes. By analyzing historical data from tournament teams spanning 2001-2024, the model identifies teams likely to succeed at each round of the tournament. While best suited as a midseason predictor, the model provides valuable metrics and insights to assess team performance and tournament potential.

## Features

- Calculates z-scores for key metrics: Adjusted Offensive Efficiency (`AdjOE`), Adjusted Defensive Efficiency (`AdjDE`), and Adjusted Efficiency Margin (`AdjEM`). For more information on these metrics, refer to https://KenPom.com.
- Identifies the likely tournament field based on `AdjEM` rankings.
- Generates probability estimates for teams advancing to each tournament round.
- Calculates implied betting odds to compare model predictions with Vegas odds.
- Exports results to a `.txt` file for easy review.

## Prerequisites

```bash
numpy>=1.21.0
pandas>=1.3.0
```

A subscription to KenPom.com is required to download historical and current season data.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/isaiahnick/ncaa-march-madness-predictive-model.git
cd ncaa-march-madness-predictive-model
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Download historical data from KenPom.com and place it in the appropriate folder.
2. Download the current season’s data from KenPom.com and BartTorvik.com. Ensure conference data is added to the dataset.
3. **Reach out to the author to access the cleaned historical dataset and update the hardcoded file path in the script.**
4. Run the main script:
```bash
python ncaaMarchMadnessModel.py
```

5. Follow the prompts to select the current season’s dataset. The script will validate the data before proceeding.

## Methodology

### Tournament Field Selection
- The top `AdjEM` team from each conference is automatically included as a qualifier.
- Remaining slots are filled by the highest-ranked teams in `AdjEM` to create a pool of 68 teams.

### Historical Threshold Analysis
- Historical data from 2001-2024 is analyzed to calculate percentile thresholds for `AdjOE`, `AdjDE`, and `AdjEM` metrics.
- Teams falling below the 10th percentile are excluded to avoid skewing results with major upsets.
- For champions, the 95th percentile is used to better identify tournament winners.
- These thresholds form pools of teams that reflect 90% of historical teams that advanced to each round, reducing the influence of outliers.

### Current Season Threshold Application
- The percentile thresholds are applied to the current dataset to identify teams most similar to successful historical teams.
- For the championship round, a stricter 95th percentile threshold is applied to focus on identifying the winner.

### Similarity Score Calculation
- Correlations between each metric and historical success (games won) are calculated to assign weights to metrics.
- For each team, the difference between the metric value and the minimum threshold is calculated.
- These differences are weighted by their normalized correlations, and the results are summed to create the final similarity score.

### Probability Normalization
- Similarity scores are transformed into probabilities for advancing through each round.
- Probabilities are constrained so that the total for all teams in a round equals the number of advancing teams (e.g., 4 for the Final Four).
- If a team’s probability exceeds 1, it is capped at 0.99, and other probabilities are scaled proportionally.

### Implied Odds Calculation
- To evaluate the model’s performance, implied Vegas odds are calculated using a standard formula and compared to the model’s probabilities.

## Outcomes and Limitations

### Outcomes
- Probability estimates and predictions for each round are exported in a `.txt` file.
- Implied odds are compared to those from sportsbooks for additional insights.
- Detailed model output is available in [ncaaModelOutcomes.xlsx](./ncaaModelOutcomes.xlsx), containing probabilities and predictions for all teams.

### Limitations
- Model accuracy is affected by extreme outliers in earlier rounds.
- Upsets are largely excluded due to reliance on percentile thresholds.
- No consideration is given to matchups, seeding, or bracket regions.
- The model’s predictive power is strongest before the bracket is released.

## Considered But Not Implemented

During development, more advanced techniques were explored, including:

- **Random Forests**: These were tested to capture non-linear relationships between metrics and success but resulted in lower initial correlations and interpretability compared to the linear approach.
- **Logistical Regression**: While useful for binary outcomes, it did not perform as well in predicting probabilities across multiple tournament rounds.
- **Non-Linear Weighting**: Attempts to introduce non-linear weighting into the similarity score calculations led to worse correlation outcomes and less transparent results.

Given these findings, the model retains a linear structure for simplicity, transparency, and superior initial outcomes.

## Future Improvements
- Integrate more metrics to refine predictions.
- Account for matchups and seeding in predictions.
- Adjust calculations to better handle outliers and early-round upsets.
- Explore machine learning approaches for enhanced predictive accuracy.

## Acknowledgments
- Historical data sourced from KenPom.com.
- Current year data sourced from KenPom.com and BartTorvik.com.

## Contact

Isaiah Nick  
Email: isaiahdnick@gmail.com
