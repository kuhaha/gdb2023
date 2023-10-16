# -*- coding:utf-8 -*-
"""
Rule-Based-PRISM
A python implementation of the PRISM algorithm for building rule based classifiers
https://github.com/dahvreinhart/Rule-Based-PRISM

Rewritten using Pandas
"""
import os
import pandas as pd

class Prism():

    def __init__(self, dataf):
        assert(os.path.exists(dataf))
        self._cvt = pd.read_csv(dataf, dtype='category')

    def fit(self, label=[]):
        data, attributes = (self._cvt.loc[:], self._cvt.columns)
        if not label:
             label = attributes[-1]
        
        # classes stores all possible values for the class
        # ex: class 'contact-lenses' can be 'none', 'soft' or 'hard', so classes = ['none','soft','hard']
        classes = self._cvt[label].cat.categories    #C
        
        # R is a list that stores all the rules
        R = []
        for cls in classes:
            # instances are the rows of the dataset
            instances = data[:]      #E
            while self.__has_class_value(instances, label, cls):
                #print("instances: " + str(len(instances)))
                rule, covered = self.__build_rule(instances, attributes, label, cls)
                #print("covered  : " + str(len(covered)))
                R.append({cls: rule})
                instances = self.__remove_covered_instances(instances, covered)

        return R, label

    def __build_rule(self, instances, attributes, label, cls):
        R, accuracy = [], -1.0
        rule_instances = instances[:]                                   
        avail_attr = [a for a in attributes if a != label]
        while True:
            allRules = []
            for A in self.__attr_not_in_R(avail_attr, R):
                for X in self.__get_attr_values(rule_instances, A):  
                    coverage = self.__apply_rule(rule_instances, [[A, X]])
                    accuracy = self.__rule_accuracy(coverage, label, cls)
                    allRules.append([A, X, accuracy[0], accuracy[1]])
                    
            bestRule = self.__get_best_rule(allRules)
            
            R.append((bestRule[0][0], bestRule[0][1]))
            
            rule_instances = self.__apply_rule(rule_instances, R)
            
            if bestRule[0][2] == 1.0 or bestRule[0][3] < 1:
                break
            
        return R, rule_instances

    def __get_best_rule(self, rules):
        rule = []
        
        runningBest = []
        maxAcc = 0
        maxCov = 0
        for potentRule in rules:
            if potentRule[2] > maxAcc:
                maxAcc = potentRule[2]
                maxCov = potentRule[3]
                runningBest.append(potentRule)
                
            elif potentRule[2] == maxAcc:
                if potentRule[3] > maxCov:
                    runningBest.append(potentRule)
                    
        rule.append(runningBest[-1])
        return rule
    
    # This method returns the instances covered by the set of rules
    def __apply_rule(self, data, R):
        coverage = data[:]
        for r in R:
            coverage = coverage[coverage[r[0]]==r[1]]
        return coverage

    # This method remove all instances covered by the set of rules
    def __remove_covered_instances(self, instances, covered):
        return instances[~instances.isin(covered)]

    # Computes p/t
    def __rule_accuracy(self, coverage, label, cls):
        if len(coverage) == 0:
             return 0.0, 0
        accuracy = coverage[coverage[label]==cls]
        return float(len(accuracy))/len(coverage), len(accuracy)

    # Counts how many instances of a given label have the specified class
    # ex: how many 'contact-lenses' = 'hard'
    def __has_class_value(self, instances, label, cls):
        if len(instances)==0:
            return 0
        asd = instances[instances[label]==cls]
        return len(asd)  # asd.shape[0]
    
    # Returns a list of all possible values of a given attribute
    def __get_attr_values(self, instances, attr):
        return instances[attr].cat.categories

    # Returns list of attributes not present in the rules
    def __attr_not_in_R(self, attr, R):
        return [a for a in attr if a not in [r[0] for r in R]]

# predict the outcome for the `case` accordings to the `rules`
def predict(case, rules):
    for rule in rules:
        k = list(rule.keys())[0]

        hit = True
        for r in rule[k]: # all predicates should be met
            if case.get(r[0]) != r[1]:
                hit = False
                break

        if hit == True:
            return k
    return 'unknown'

# Debugging function printing the set of rules in english
def printRules(rules, label):
    for rule in rules:
        k = list(rule.keys())[0]
        nbr = len(rule[k])
        
        theRule = " IF "
        for subrules in rule[k]:
            nbr -= 1
            theRule = theRule + str(subrules[0]) + " = " + subrules[1]
            if nbr > 0:
                theRule = theRule + "\n\t AND "
            else:
                theRule = theRule + "\n\t THEN " + label + " = " + k
        
        print (theRule +"\n")


if __name__ == '__main__':
    import sys
    prism = Prism(sys.argv[1])
    rules, label = prism.fit()
    
    printRules(rules,label)
    #print (rules)
    print ("-*" * 20)
    case = {'astigmatism': 'yes', 'tear-prod-rate': 'normal'}
    result = predict(case, rules)
    print (case)
    print ("recommended lens: " + result)

    print ("-*" * 20)
    case = {'astigmatism': 'yes', 'tear-prod-rate': 'reduced'}
    result = predict(case, rules)
    print (case)
    print ("recommended lens: " + result)
    print ("-*" * 20)