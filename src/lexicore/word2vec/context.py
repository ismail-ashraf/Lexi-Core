class ContextTargetGenerator:

    def __init__(self , context_window):
        self.context_window = context_window


    def generate(self,tokens):
        pairs = []

        for i in range(len(tokens)):
            target = tokens[i]

            start = max(0 , i - self.context_window)

            end = min(len(tokens) , i + self.context_window + 1)

            for j in range(start,end):
                if j == i:
                    continue

                else:
                    pairs.append((target,tokens[j]))

        return pairs



# tokens = [2, 5, 8, 11]
#
# generator = ContextTargetGenerator(context_window=1)
#
# pairs = generator.generate(tokens)
#
# print(pairs)
# print(tokens)