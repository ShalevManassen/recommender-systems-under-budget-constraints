import numpy as np
from itertools import combinations
import time


class Recommender:
    def __init__(self, n_weeks: int, n_users: int, prices: np.array, budget: int):
        # Store the exact creation timestamp of the object to track its lifetime
        self.created_at = time.time()

        self.n_rounds = n_weeks
        self.n_users = n_users
        self.item_prices = prices
        self.budget = budget

        # Number of items (podcasts)
        self.mum_podcasts = len(self.item_prices)

        # All valid subsets of item indices that fit within the budget
        self.valid_sub_groups = self.all_possible_sub_groups()

        # Current round index (used for warm-up phase)
        self.curr_k = 0

        # Initialize tracking matrices for user-item statistics
        self.means = np.zeros((self.n_users, self.mum_podcasts))  # Estimated means
        self.nta = np.ones((self.n_users, self.mum_podcasts))  # Number of trials (starts with 1 to avoid divide by 0)
        self.sum_results = np.zeros((self.n_users, self.mum_podcasts))  # Accumulated rewards
        self.UCB_per_ui = np.zeros((self.n_users, self.mum_podcasts))  # UCB values

        # Last recommendation made for each user
        self.last_recommend = np.zeros(self.n_users)

        # Parameter of UCB
        self.UCB_parameter = 0.1

    # Check if 110 seconds have passed since the object was created
    def is_expired(self):
        return (time.time() - self.created_at) >= 110

    # Returns all subgroups of item indices whose total price is within budget
    def all_possible_sub_groups(self):
        n = self.mum_podcasts
        valid_groups = []
        for r in range(1, n + 1):
            for idx in combinations(range(n), r):
                group = list(idx)
                group_price = np.sum(self.item_prices[group])
                if group_price <= self.budget:
                    valid_groups.append(group)
        return np.array(valid_groups, dtype=object)

    # Update UCB scores only for the items that were last recommended to each user
    def UCB_calculate(self):
        for i in range(self.n_users):
            j = self.last_recommend[i]  # last recommended item index for user i
            confidence = (self.UCB_parameter * np.log(self.n_rounds) / self.nta[i][j]) ** 0.5
            self.UCB_per_ui[i][j] = self.means[i][j] + confidence

    # Update means and counts only for the items just recommended
    def update_means(self, results):
        for i in range(self.n_users):
            j = self.last_recommend[i]
            self.nta[i][j] += 1
            self.sum_results[i][j] += results[i]
            self.means[i][j] = self.sum_results[i][j] / self.nta[i][j]

    # Find the subgroup of items (indices) that maximizes the average best-UCB across users
    def best_sub_group(self) -> np.array:
        best_score = -np.inf
        best_group = None
        best_recommendation = None

        for group in self.valid_sub_groups:
            group = np.array(group, dtype=int)
            sub_ucb = self.UCB_per_ui[:, group]  # UCB scores restricted to this group

            best_indices = np.argmax(sub_ucb, axis=1)  # Best item per user within group

            score = np.average(sub_ucb[np.arange(self.n_users), best_indices])  # Average UCB across users

            if score > best_score:
                best_score = score
                best_group = group
                # Map best indices (relative to group) back to item indices
                best_recommendation = np.array([group[i] for i in best_indices])

        return best_recommendation

    # Generate recommendations
    def recommend(self) -> np.array:
        # If the object has expired, return the last recommendation
        if self.is_expired():
            return self.last_recommend
    # Warm-up phase: recommend each item once to all users
        # Skipping items with price bigger then budget
        while ((self.curr_k < self.mum_podcasts) and (self.item_prices[self.curr_k] > self.budget)):
            self.curr_k += 1
        if self.curr_k < self.mum_podcasts:
            recommend_array = np.full(self.n_users, self.curr_k)
            self.curr_k += 1
            self.last_recommend = recommend_array
            return recommend_array

        # After warm-up: recommend using UCB strategy on best subgroup
        else:
            recommend_array = self.best_sub_group()
            self.last_recommend = recommend_array
            return recommend_array

    # Update statistics based on received rewards
    def update(self, results: np.array):
        # If the object has expired, return the last recommendation
        if self.is_expired():
            return self.last_recommend
        # Warm-up: update with full user responses for current item
        if self.curr_k <= self.mum_podcasts:
            self.sum_results[:, self.curr_k - 1] = results
            self.means[:, self.curr_k - 1] = results
            self.UCB_calculate()
            if (self.curr_k == self.mum_podcasts):
                self.curr_k += 1

        # After warm-up: update UCB values according to new results
        else:
            self.update_means(results)
            self.UCB_calculate()
