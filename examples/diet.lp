\ Stigler style diet problem: cheapest daily menu meeting nutrient minimums.
\ Pure LP, so the CLI also prints shadow prices: the marginal cost of
\ one more unit of each nutrient requirement.
Minimize
 cost: 0.36 oats + 0.85 milk + 1.20 eggs + 0.48 bread + 2.10 chicken + 0.60 beans
Subject To
 calories: 150 oats + 120 milk + 155 eggs + 80 bread + 240 chicken + 130 beans >= 2000
 protein:  5 oats + 8 milk + 13 eggs + 3 bread + 27 chicken + 9 beans >= 65
 calcium:  20 oats + 300 milk + 50 eggs + 30 bread + 15 chicken + 40 beans >= 800
 iron:     2 oats + 0 milk + 1.2 eggs + 1 bread + 1.3 chicken + 2.5 beans >= 12
Bounds
 oats <= 4
 eggs <= 3
 bread <= 6
End
