class Inventory:
    def __init__(self, items=[]):
        self.items = items

    def add(self, item):
        self.items.append(item)
        return self
