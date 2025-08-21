# personalized-podcast-recommendation

## Project Description
Given a new podcast platform with limited budget for producing episodes, the goal is to design a **recommender system** that maximizes user enjoyment over multiple weeks. 
Each week, the system must decide which podcasts to produce (subject to production costs and a budget constraint) and which podcast to recommend to each user. 
User satisfaction is binary (enjoy / not enjoy) and unknown in advance, so the system must learn from past feedback. The challenge is to balance **exploration and exploitation** under budget limits while optimizing total user enjoyment.

## Project Input (csv files)
- **prices**: A list of podcast production costs.  
- **budget**: The weekly budget constraint.  
- **P (probability matrix)**: User–podcast enjoyment probabilities (hidden during simulation).  
- **n_weeks, n_users**: Number of weeks and number of users in the simulation.
