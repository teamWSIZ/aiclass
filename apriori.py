from apyori import apriori

"""
Support = popularity of item
Confidence  = likelihood that an item B is also bought if item A is bought
Confidence(A→B) = (Transactions containing both (A and B))/(Transactions containing A)

Lift(A -> B) refers to the increase in the ratio of sale of B when A is sold.

Lift(A→B) = (Confidence (A→B))/(Support (B))

"""

transactions = [
    ['A','B'],
    ['A','B'],
    ['A','B'],
    ['A','B'],
    ['A','B'],
    ['a','B'],
    ['A'],
    ['A','b'],
]
results = list(apriori(transactions, min_support=0.2, min_confidence=0.2, min_lift=0.2))

for item in results:

    # first index of the inner list
    # Contains base item and add item
    if len(item.items) < 2:
        continue
    # print(len(item.items))

    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])
    #
    # second index of the inner list
    print("Support: " + str(item[1]))

    # third index of the list located at 0th
    # of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
