import numpy as np  
import math
import random
import trimesh                      # Mesh for wrench space
from stl import mesh                # Mesh for merging friction cone
import vtkplotlib as vpl            # Mesh visualization   
from numpy import linalg as LA      # Norm of a vector
from scipy.optimize import linprog  # Linear Programming interface
from scipy.spatial import Delaunay  # Convex Hull generation

'''

Input : stl file of an object

Output:

1. Object with friction cones visualized
2. Force-closure grasp determination within random antipodal grasp points

'''
def unit(x):
    div = math.sqrt(sum(x*x))
    y = x/div
    return y

stl_name = 'nut'

# Load the mesh to remove inner mesh grasp candidates
mesh_tri = trimesh.load('./objects/' + stl_name +'.stl')
mesh_tri.vertices -= mesh_tri.center_mass
mesh_tri.export('./objects/' + stl_name +'_modified.stl')

ver = mesh_tri.vertices
tri = mesh_tri.triangles
# Each value of triangle array in 'tri' refers to the index of vertex in 'ver'.

# Params

unit_contact_force = 1
K   = - unit_contact_force  # Normal vector of each triangle is basically headed outside (tri.face_normals()) 


m   = 0.5      # static friction coefficient
m_n = 0.5      # torsional friction coefficient


W_1 = np.zeros([1,2,3])
W_2 = np.zeros([1,2,3])
W_3 = np.zeros([1,2,3])
W_4 = np.zeros([1,2,3])
T   = np.zeros([1,3])

# Torque multiplier 1 (GraspIt!)
contact_points = np.array(ver)
dists = np.sqrt(np.einsum('ij,ij->j', contact_points.T, contact_points.T))
rho = np.max(dists)

'''
Torque multiplier 2 
[Roa, M. A., & Suárez, R. (2009). Computation of independent contact regions for grasping 3-d objects. 
 IEEE Transactions on Robotics, 25(4), 839-850.]

for i in range(ver.shape[0]):
    dist = ver[i]
    rho += sum(dist*dist) 
rho2 = math.sqrt(rho / ver.shape[0])
'''

for i in range(tri.shape[0]): 
    
    # Each vertex in i th triangle
    e1 = tri[i][0]
    e2 = tri[i][1]
    e3 = tri[i][2]
    e4 = mesh_tri.triangles_center[i]  # center of the mesh, which is the origin of object frame

    # ith triangle in {object}
    x = unit(e3-e2)
    z = mesh_tri.face_normals[i]
    y = unit(np.cross(z,x))

    # Homogeneous transformation matrix from {object} to {world}
    x = x.reshape(3,1)
    y = y.reshape(3,1)
    z = z.reshape(3,1)
    e4 = e4.reshape(3,1)
    
    r = np.hstack((x,y,z))  # Homogeneous transformation matrix from {object} to {world}
    rr = np. linalg. inv(r) # Homogeneous transformation matrix from {world} to {object}
    
    # Normal force onto the center of i th triangle
    
    # Frictional cone (linearly approximated to rectangular cone)
    f_n = K / math.sqrt(1 + m**2)            # Magnitude of normal force
    f_t = K * m                              # Magnitude of tangential force

    # Forces in {object}
    f   = np.array([  0 ,  0 ,  K ]).reshape(1,3) 
    f_1 = np.array([f_t ,0   , f_n]).reshape(1,3)
    f_2 = np.array([0   ,f_t , f_n]).reshape(1,3)
    f_3 = np.array([-f_t,0   , f_n]).reshape(1,3)
    f_4 = np.array([0   ,-f_t, f_n]).reshape(1,3)
    
    # Forces in {world} 
    f_1 = np.matmul(r,f_1.T).T
    f_2 = np.matmul(r,f_2.T).T
    f_3 = np.matmul(r,f_3.T).T
    f_4 = np.matmul(r,f_4.T).T

    # Torques in {world}
    torque_1 = np.cross(e4.reshape(1,3),f_1)
    torque_2 = np.cross(e4.reshape(1,3),f_2)
    torque_3 = np.cross(e4.reshape(1,3),f_3)
    torque_4 = np.cross(e4.reshape(1,3),f_4)

    # Torsion in {world} [Soft contact model]
    torsion   = f_n * m_n * mesh_tri.face_normals[i][np.newaxis] # world frame
         
    # Wrenches in {world}
    w_1 = np.append(f_1  ,torque_1 /rho,  axis = 0)
    w_2 = np.append(f_2  ,torque_2 /rho,  axis = 0)
    w_3 = np.append(f_3  ,torque_3 /rho,  axis = 0)
    w_4 = np.append(f_4  ,torque_4 /rho,  axis = 0)
    
    # Stack tf matrix and wrench into the i th index
    W_1 = np.append(W_1,w_1.reshape(1,2,3), axis = 0)
    W_2 = np.append(W_2,w_2.reshape(1,2,3), axis = 0)
    W_3 = np.append(W_3,w_3.reshape(1,2,3), axis = 0)
    W_4 = np.append(W_4,w_4.reshape(1,2,3), axis = 0)
    
    T   = np.append(T,torsion, axis = 0)
    
# Resultant wrench
W_1 = W_1[1:]
W_2 = W_2[1:]
W_3 = W_3[1:]
W_4 = W_4[1:]
T   =  T [1:]

# Add Friction cones to object mesh for visualization

# Object mesh for visualization
main_body = mesh.Mesh.from_file('./objects/' + stl_name+'_modified.stl')

scale = -0.5
for i in range(W_1.shape[0]):
    e4 = mesh_tri.triangles_center[i][np.newaxis]

    f_1 = W_1[i,0][np.newaxis]* scale
    f_2 = W_2[i,0][np.newaxis]* scale
    f_3 = W_3[i,0][np.newaxis]* scale
    f_4 = W_4[i,0][np.newaxis]* scale
    
    f_1 = e4 + f_1
    f_2 = e4 + f_2
    f_3 = e4 + f_3
    f_4 = e4 + f_4

    # Define the vertices composing the frictional cone
    vertices = np.concatenate((e4, f_1, f_2, f_3, f_4),axis=0)
    
    # Define the triangles composing the frictional cone
    faces = np.array([\
        [0,1,2],
        [0,2,3],
        [0,3,4],
        [0,4,1],
        [1,4,3],
        [1,3,2],])

    # Create the mesh
    cone = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for k, f in enumerate(faces):
        for j in range(3):
            cone.vectors[k][j] = vertices[f[j]]

    # Merge friction cones to the object
    main_body = mesh.Mesh(np.concatenate([main_body.data, cone.data]))


# main_body.save('./objects/' + stl_name +'_with_friction_cones.stl')  # save as ASCII

# Visualize the object
fig = vpl.figure()
mesh = vpl.mesh_plot(main_body)
vpl.show()

# Find random antipodal grasp points
points = np.array(random.sample(list(range(tri.shape[0])),2))

p1 = points[0]
p2 = points[1]

# Wrenches of the first grasp point

w11 = W_1[p1].reshape(1,6)[0]
w12 = W_2[p1].reshape(1,6)[0]
w13 = W_3[p1].reshape(1,6)[0]
w14 = W_4[p1].reshape(1,6)[0]

w15 =  np.append(np.zeros([1,3]),T [p1][np.newaxis],axis=1).reshape(1,6)[0]

## Wrenches of the second grasp point

w21 = W_1[p2].reshape(1,6)[0]
w22 = W_2[p2].reshape(1,6)[0]
w23 = W_3[p2].reshape(1,6)[0]
w24 = W_4[p2].reshape(1,6)[0]

w25 =  np.append(np.zeros([1,3]),T [p2][np.newaxis],axis=1).reshape(1,6)[0]

## 6D Convex Hull of wrenches

wrench_map = np.zeros([1,6])
b = 1
while b < 6: 
    exec(f'wrench_map = np.append(wrench_map,w1{b}[np.newaxis],axis = 0)')
    exec(f'wrench_map = np.append(wrench_map,w2{b}[np.newaxis],axis = 0)')
    b+=1

wrench_map = wrench_map[1:]  


P = np.sum(wrench_map,axis=0) / wrench_map.shape[0] # Point P: Centroid of wrench_map
ray = -P                                            # Vector of ray shot from P to O (origin of Convex Hull)
hull = Delaunay(wrench_map)                         # CH(W)
        
ver_index = hull.convex_hull                        # Vertex indices of each hyperplane of CH
point     = hull.points                             # Inner points of CH

## Hyperplanes of corresponding Convex Polytope
CH_vertices  = np.zeros([1,6])
offset = np.ones([1,6])

# Get vertices of Convex Hull
CH_ver_index=[] 
for i in range(ver_index.shape[0]):
    for j in range(ver_index.shape[1]):
        
        CH_ver_index.append(ver_index[i,j])
        
CH_ver_index = list(set(CH_ver_index)) # Indices of vertices of CH(W)

for i in range(len(CH_ver_index)):
    
    CH_vertices = np.append(CH_vertices, point[i][np.newaxis] - P, axis=0)
    
CH_vertices = CH_vertices[1:] # Vertices of CH(W) with translation of P to origin = Coefficients of hyperplanes of corresponding Convex Polytope

offset = np.ones([CH_vertices.shape[0],1])

# Execute Linear Programming through Simplex Algorithm
ans = linprog(ray, A_ub = CH_vertices,  b_ub = offset,  method='simplex')
E = ans.x # Optimal point E

if ans.status == 0:
    print("Optimization terminated successfully.")
    print("Optimal Point: ", ans.x)


Q = P/np.dot(E,P) # Intersection btw ray and Convex Hull
PQ = LA.norm(P-Q) # Length of Line Segment PQ
PO = LA.norm(P)   # Length of Line Segment PO

print("Length of Line Segment PQ:", PQ)
print("Length of Line Segment PO:", PO)

# Determine force-clousre grasp through length comparison of PQ and PO
if PQ > PO:
    print("It is a force-closure grasp.")
else:
    print("It is not a force-closure grasp.")