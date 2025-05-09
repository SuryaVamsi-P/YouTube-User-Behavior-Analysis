## YouTube Video Analytics Dashboard

## Overview  

This project dives deep into **YouTube trending video analytics** through both **static visualization** and an **interactive dashboard**. Using raw `youtube.csv` data, the analysis explores how **video performance metrics (views, likes, comments)** relate to **time of publication**, **country**, and **channel** behavior.

The objective was to **extract insights that guide content strategy**, optimize publishing time, and understand audience engagement patterns across different categories and geographies.


## Key Features

- **Statistical Analysis & Preprocessing**: 
  - Outlier detection & treatment using IQR and Box-Cox transformation
  - Normality testing using Kolmogorov-Smirnov test
  - Correlation analysis via Pearson heatmaps

- **Data Cleaning & Feature Engineering**: 
  - Cleaned & transformed 14+ raw features (e.g. tags, publish time, part of day)
  - Generated new attributes like `part_of_day`, `weekday`, `publish_country`, etc.

- **Static Visualizations (Matplotlib & Seaborn)**:  
  - Histograms, Boxplots, Heatmaps, Line Charts, Pie Charts, Contour Plots

- **Interactive Dashboard (Dash & Plotly)**:
  - Channel-level metrics: Views, Likes, Comments, Dislikes
  - Time of day vs day-of-week publishing impact
  - Country-wise distribution and trends
  - Filters for year, country, and custom checklists


- **Tech Stack**:  
  `Python`, `Pandas`, `NumPy`, `Seaborn`, `Matplotlib`, `Plotly`, `Dash`, `Scikit-learn`, `PrettyTable`


## Skills Demonstrated

- Data wrangling & EDA  
- Dimensionality Reduction (PCA)  
- Dashboard development  
- Data storytelling with plots  
- Insights generation for strategic planning


## Repository Structure

```
YouTube-Analytics-Dashboard
├── FTP(G40559527).py                     # Full static analysis + plots
├── Youtube Dashboard.py                  # Dash application for interactivity
├── youtube.csv                           # Dataset
├── Youtube Video Analysis(REPORT).pdf    # Summary report of project
├── README.md                             # Project overview (this file)
```


## Insights Gleaned

- **Afternoon videos gain higher engagement** across most categories.
- Channels with consistent posting on **weekends outperform** others in views.
- Certain countries show skewed like/comment ratios indicating regional behavior.


## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/youtube-analytics-dashboard.git
cd youtube-analytics-dashboard

# 2. Install requirements
pip install -r requirements.txt

# 3. Run the Dash App
python 'Youtube Dashboard.py'
```

## Let's Connect

**Author**: Surya Vamsi Patiballa  
**MS in Data Science**, George Washington University  
svamsi2002@gmail.com  
[My Resume] https://drive.google.com/file/d/19IKd1OQ20OhHBJkTkVk7T6zIfQ7EFibZ/view?usp=sharing

[LinkedIn] https://www.linkedin.com/in/surya-patiballa-b724851aa/
