# ForceClosureDeterminator
Determinates whether the given grasp is force closure

Input: STL file of an object
Output: 
- Object with friction cones corresponding to possible grasp points (1)
- Determination of force closure of a random 2-finger grasp (2)

Algorithm runs as follows:
1. Get possible grasp points according to object mesh triangles (line 33)

2. Calculate grasp wrench (from the friction cone) for each grasp point (line 133)
Friction cone is linearized to a rectangular cone
  Assume:
    1) Soft finger contact model
    2) Unit normal force
    3) Torque multiplier of GraspIt! definition

3. Visualize friction cones merged with object mesh (line 184)
4. Find a pair of random antipodal grasp points (line 189)
5. Draw corresponding wrench (line 195)
6. Calculate convex hull and corresponding convex polytope (line 227,249)
7. Find the optimal point through LP-simplex (line 253)
8. Find the intersecting point of a ray and a facet of convex hull (line 262)
9. Determine force-closure grasp (line 269)


<div align="center">

![image](https://github.com/Hongyoungjin/ForceClosureDeterminator/assets/69029439/1de42e17-eba3-42db-8467-1811509c111b)

</div>
