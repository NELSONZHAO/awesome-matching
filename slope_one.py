"""
@author: nelson zhao
@date: 2020-03-07

"""
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


class SlopeOneMatch(object):
    def __init__(self):
        self.data = None

        self.user_col = None
        self.item_col = None
        self.rating_col = None

        self.uids = None
        self.iids = None

        self.user_item_index = {}
        self.item_cooccurrence = {}
        self.item_rating_delta = {}

    def __build_user_inverted_index(self, uids):
        """
        Build user-item inverted index
        :return:
        """
        user_item_index = {}
        for _, row in data[data[self.user_col] == uids].iterrows():
            user = row[self.user_col]
            item = row[self.item_col]
            rating = row[self.rating_col]

            user_item_index.setdefault(user, {})
            user_item_index[user].setdefault(item, 0)
            user_item_index[user][item] = rating

        return user_item_index

    def build_user_inverted_index(self):
        """
        Build user-item inverted index fast
        :return:
        """
        self.uids = self.data[self.user_col].unique()

        with ThreadPoolExecutor(max_workers=8) as executor:
            rlt = executor.map(self.__build_user_inverted_index, self.uids)

        for batch in rlt:
            self.user_item_index.update(batch)

    def build_item_delta_rating(self):
        for _, item_ratings in self.user_item_index.items():
            item_size = len(item_ratings)
            item_ratings = list(item_ratings.items())

            for i in range(item_size):
                for j in range(i + 1, item_size):
                    item_i = item_ratings[i][0]
                    item_j = item_ratings[j][0]

                    rating_i = item_ratings[i][1]
                    rating_j = item_ratings[j][1]
                    item_delta = rating_i - rating_j

                    self.item_cooccurrence.setdefault(item_i, {})
                    self.item_cooccurrence[item_i].setdefault(item_j, [0, 0])
                    self.item_cooccurrence[item_i][item_j][0] += item_delta
                    self.item_cooccurrence[item_i][item_j][1] += 1

                    self.item_cooccurrence.setdefault(item_j, {})
                    self.item_cooccurrence[item_j].setdefault(item_i, [0, 0])
                    self.item_cooccurrence[item_j][item_i][0] += -item_delta
                    self.item_cooccurrence[item_j][item_i][1] += 1

    def fit(self, data):
        """
        fit model given training data
        :param data: training data
        :return:
        """
        self.data = data
        assert self.data.shape[-1] >= 3, "The input data must have three columns: user, item, rating"
        cols = data.columns.values
        self.user_col = cols[0]
        self.item_col = cols[1]
        self.rating_col = cols[2]

        # user item map
        self.build_user_inverted_index()
        # item cooccurrence ratings
        self.build_item_delta_rating()

    def predict(self, user, item):
        assert user in self.user_item_index, "user cannot be found."
        weighted_rating_delta = 0
        weights = 0

        items_ratings = self.user_item_index[user]
        if item in items_ratings:
            return items_ratings[item]

        for i, r in items_ratings.items():
            delta, cnt = self.item_cooccurrence[i].get(item, [0, 0])
            weighted_rating_delta += cnt * r - delta
            weights += cnt

        rating = weighted_rating_delta / weights if weights > 0 else -1
        return rating

    def match(self, user, candidates, top_k=10):
        assert user in self.user_item_index, "user cannot be found."
        assert len(candidates) > 0, "The number of item candidates must be large than 0."

        match_items = []
        item_ratings = self.user_item_index[user]
        for candidate in candidates:
            if candidate in item_ratings:
                continue
            pred_rating = self.predict(user, candidate)
            match_items.append((candidate, pred_rating))

        match_items.sort(key=lambda x: x[1], reverse=True)
        return match_items[:top_k]

    def get_user_item_index(self):
        return self.user_item_index

    def get_item_cooccurrence(self):
        return self.item_cooccurrence


if __name__ == "__main__":

    data = pd.read_table("./datasets/ratings.dat", sep="::", nrows=10000, header=None)
    data.columns = ['user', 'item', 'rating', 'ts']
    slope_one_match = SlopeOneMatch()
    slope_one_match.fit(data)

    items = data['item'].unique()
    rlt = slope_one_match.match(1, items, 100)
    print(rlt)
