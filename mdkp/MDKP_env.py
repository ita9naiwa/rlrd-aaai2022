# import gym
import numpy as np
import copy

EPS = 1e-8


class Item(object):
    """
    A class for single category item.
    """
    def __init__(self, weights, num_items, value):
        """
        Args:
            weights : [dimension] numpy row vector.
            num_items : There can be num_items many identical items. (3 identical book, ...)
            value : An integer
        """
        self.weights = np.array(weights)
        self.amount = num_items
        self.remained_amount = num_items        # How many items remained?
        self.value = value

    def as_vector(self):
        return np.hstack([self.value, self.weights]).reshape(1, -1)


class Knapsack(object):
    def __init__(self, capacity):
        """
        Args:
            capacity : d - dimensional numpy row vector.
        """
        self.capacity = np.array(capacity)
        self.res_capacity = copy.deepcopy(self.capacity)
        self.stored_item = []

    def insert(self, item):
        self.res_capacity -= item.weights
        self.stored_item.append(item)

    def as_vector(self):
        return self.capacity

    def check_allocability(self, item):
        ok = item.remained_amount >= 1 and np.all(
            self.res_capacity >= item.weights
        )
        return ok

    def calc_util(self, item):
        return np.mean((item.weights / (self.res_capacity + EPS)))

    def calc_value(self):
        # Return the sum of value of items stored in knapsack.
        ret = 0
        for item in self.stored_item:
            ret += item.value
        return ret

    def get_weights(self):
        ret = np.zeros_like(self.capacity)
        for item in self.stored_item:
            ret += item.weights
        return ret


class Env(object):
    def __init__(
            self, dimension, num_items,
            num_knapsacks, w, c
    ):
        """
        Args:
            dimension : How many dimension you consider.
            num_items : The number of categories of items. [num_items] numpy vector.
            weights : the weights of each items. [num_items x dimension]
            values : the values of each items. [num_items vector]
        EX:
            Three books, Five Keyboards, Two Dogs
            - > num_items = 3, per_num_items = [3, 5, 2] // a list.
        """
        super(Env, self).__init__()
        self.dimension = dimension
        self.num_knapsacks = num_knapsacks
        self.w = w
        self.a = 0.5
        self.num_items = num_items
        self.knapsacks = []
        self.items = []
        self.rewards = []
        self.c = c

    def check_allocability(self, item, knapsack):
        return item.remained_amount >= 1 and np.all(knapsack.res_capacity >= item.weights)

    def get_reward(self, item_idx, knapsack_idx):
        return 0
        # item = self.items[item_idx]
        # knapsack = self.knapsacks[knapsack_idx]
        # return self.values[item_idx] / (knapsack.calc_util(item)+EPS)

    def get_allocable_items(self):
        """
        Return the list of integers i,
        where ith item is allocable
        """
        ret = []
        for i, item in enumerate(self.items):
            for j, knapsack in enumerate(self.knapsacks):
                ok = knapsack.check_allocability(item)
                if ok:
                    ret.append(i)
                    break
        return ret

    def get_allocable_knapsack(self):
        """
        Three items --> Return : [[0, 1], [], [1]]
        : First item can be assigned to 0, 1st knapsack,
          Second item cant be assigned,
          Third one assigned only to 1st knapsack.
        """
        ret = []
        for i, item in enumerate(self.items):
            possible_knapsack = []
            for j, knapsack in enumerate(self.knapsacks):
                ok = knapsack.check_allocability(item)
                if ok:
                    possible_knapsack.append(j)
            ret.append(possible_knapsack)
        return ret

    def is_done(self):
        x = self.get_allocable_items()
        return len(x) == 0

    def as_vector(self):
        items = np.vstack([item.as_vector() for item in self.items])
        knapsacks = np.vstack([knapsack.as_vector() for knapsack in self.knapsacks])
        return items, knapsacks

    def observe(self):
        items, knapsack = self.as_vector()
        return items, knapsack, self.get_allocable_items(), self.get_allocable_knapsack()

    def reset(self, knapsack_capacities=None,
              per_num_items=None, weights=None, values=None):

        self.items = []
        self.knapsacks = []

        self.per_num_items = per_num_items
        if per_num_items is None:  # Just one. 휴리스틱땜에 이렇게 해야함
            self.per_num_items = np.random.randint(1, 2, self.num_items)

        self.weights = np.random.randint(1, self.w, (self.num_items, self.dimension))
        self.values = self.c * np.mean(self.weights, -1) + (1 - self.c) * np.random.randint(1, 200)

        self.knapsack_capacities = self.a * np.sum(self.weights, axis=0)
        self.knapsack_capacities = self.knapsack_capacities.reshape(1, -1)

        for i in range(self.num_items):
            self.items.append(
                Item(self.weights[i], self.per_num_items[i], self.values[i])
            )
        for i in range(self.num_knapsacks):
            self.knapsacks.append(Knapsack(self.knapsack_capacities[i]))

        self.rewards = []

        return self.observe(), self.per_num_items

    def step(self, item_idx, knapsack_idx):
        """
        Args:
            item_idx : An integer! Not a item class. This is output of our policy network.
            knapsack_idx : An integer. if item_idx = 3, knapsack_idx = 7, then 3rd item
                           goes into the 7th knapsack.
        """
        item = self.items[item_idx]

        if item.remained_amount <= 0:
            raise EnvironmentError("Why empty item can be assigned?")

        self.knapsacks[knapsack_idx].insert(item)
        item.remained_amount -= 1
        reward = self.get_reward(item_idx, knapsack_idx)
        self.rewards.append(reward)

        if self.is_done():
            self.rewards[-1] += self.total_value()
            reward += self.total_value()
        return self.observe(), reward, self.is_done()

    def total_value(self):
        """
        Return:
             The sum of values of items in knapsack by far.
        """
        ret = 0
        for k, knapsack in enumerate(self.knapsacks):
            ret += knapsack.calc_value()
        return ret
