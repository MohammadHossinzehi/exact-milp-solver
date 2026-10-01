\ Capacitated facility location: open warehouses (binary o1..o3, fixed cost)
\ and ship goods (continuous s_ij) to three customers at minimum total cost.
Minimize
 cost: 400 o1 + 350 o2 + 500 o3 + 4 s11 + 6 s12 + 9 s13 + 5 s21 + 4 s22 + 7 s23 + 6 s31 + 3 s32 + 4 s33
Subject To
 d1: s11 + s21 + s31 = 80
 d2: s12 + s22 + s32 = 70
 d3: s13 + s23 + s33 = 90
 cap1: s11 + s12 + s13 - 150 o1 <= 0
 cap2: s21 + s22 + s23 - 120 o2 <= 0
 cap3: s31 + s32 + s33 - 200 o3 <= 0
Binary
 o1 o2 o3
End
