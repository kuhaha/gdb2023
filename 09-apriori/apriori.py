from apyori import apriori, load_transactions

with open('basket_data.csv') as f:
    transactions = load_transactions(f, delimiter=",")

    association_rules = apriori(transactions,
             min_support=0.3,
             min_confidence=0.80,
             min_lift=1.0,
             max_length=None)

    #print (list(association_rules))  #dump raw data for debugging
    
    for ruleset in association_rules:
        items = [x for x in ruleset.items]
        print (str(items))
        support = ruleset.support
        rules = ruleset.ordered_statistics
        for rule in rules:
            lhs =  str(list(rule.items_base)) # left-hand side
            rhs =  str(list(rule.items_add))  # right-hand side
            print("Rule: " + lhs + " -> " + rhs)
            print("Support: " + str(support))
            print("Confidence: " + str(rule.confidence))
            #print("Lift: " + str(rule.lift))
            print("=====================================")
