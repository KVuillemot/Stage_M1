import dolfin as df 
import matplotlib.pyplot as plt 
import mshr
import time
import numpy as np 
from vedo.dolfin import plot, interactive, screenshot, closePlotter 
# dolfin parameters
df.parameters["ghost_mode"] = "shared_facet" 
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters['allow_extrapolation'] = True
df.parameters["form_compiler"]["representation"] = 'uflacs'

# degree of interpolation for V and Vphi
degV = 1
degPhi = 1 + degV

# creation of the expression of phi 
class phi_expr(df.UserExpression) :
    def closest(self, x):
        # we compute the two closests points to x that are on the scatter plot
        num_cell = boundary_mesh_tree.compute_closest_entity(df.Point(x))[0]
        if num_cell>= 0 and num_cell <= boundary_mesh.num_cells() :
            c = df.Cell(boundary_mesh,num_cell)
            p1x,p1y,p2x,p2y = c.get_coordinate_dofs()
            return np.array([p1x,p1y]), np.array([p2x,p2y]) 
        else :
            min1 = 10.0
            for point in points :
                dist = np.linalg.norm(x-point)
                if dist < min1 :
                    min1 = dist
                    close1 = point
            min2 = 10.0
            for point in points :
                dist = np.linalg.norm(close1-point)
                if dist < min2 and dist != 0.0 :
                    min2 = dist
                    close2 = point
            return close1, close2
            
    def project(self, x):
        # we compute the projection of x on the segment [p1, p2] after checking if x is on the scatter plot
        p1, p2 = self.closest(x)
        if len(boundary_mesh_tree.compute_collisions(df.Point(x[0],x[1]))) != 0 : 
            proj_x = x 
        else :   
            p1x = x-p1 
            p1p2 = (p2-p1)/np.linalg.norm(p2-p1)
            proj_x = p1 + (p1x@p1p2)*p1p2
        return proj_x
    
    def inside(self, x):
        # we check if x is inside or outside the domain (intersection between x and one entity of the mesh)
        return (len(tree.compute_collisions(df.Point(x[0],x[1]))) != 0) # return false if outside true if inside

    def eval(self, value, x):
        # we compute the projection 
        proj_x = self.project(x)
        # then we take the signed distance between x and the projection        
        if self.inside(x):
            value[0] = -np.linalg.norm(x-proj_x)
        else :
            value[0] = np.linalg.norm(x-proj_x)
        
    def value_shape(self):
        return (2,) 

class phi_expr_ex(df.UserExpression) : 
    def eval(self, value, x):
        value[0] = - df.sqrt(1.0/8.0) + df.sqrt((x[0]-0.5)**2 + (x[1]-0.5)**2)
    def value_shape(self):
        return (2,)

# We create the lists that we'll use to store errors and computation time for the phi-fem and standard fem
Time_assemble_phi, Time_solve_phi, Time_total_phi, error_l2_phi, error_h1_phi, hh_phi, nbr_points = [], [], [], [], [], [], []
Time_assemble_standard_ex, Time_solve_standard_ex, Time_total_standard_ex, error_h1_standard_ex, error_l2_standard_ex,  hh_standard_ex = [], [], [], [], [], []

start,end,step = 1,5,1

# Computation of the standard fem with exact expression of phi
domain = mshr.Circle(df.Point(0.5,0.5),df.sqrt(2.0)/4.0) # creation of the domain
for i in range(start, end, step):
    H = 10*2**i # to have approximately the same precision as in the phi-fem computation
    mesh = mshr.generate_mesh(domain,H)
    print("Standard fem iteration : ", i)
    # FunctionSpace P1
    V = df.FunctionSpace(mesh, 'CG', degV)  
    v = df.TestFunction(V)
    u = df.TrialFunction(V)
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr_ex(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    u_ex = df.Expression("exp(x[0])*sin(2*pi*x[1])",degree=4,domain=mesh)
    u_D = df.Expression("exp(x[0])*sin(2*pi*x[1])",degree=4,domain=mesh) * (1 + phi)
    f = -df.div(df.grad(u_ex))
    # Definition of the boundary to apply Dirichlet condition
    def boundary(x,on_boundary):
        return on_boundary
    bc = df.DirichletBC(V, u_D, boundary)
    # Resolution of the variationnal problem
    a = df.inner(df.grad(u),df.grad(v))*df.dx
    L = f*v*df.dx
    start_assemble = time.time()
    A = df.assemble(a)
    B = df.assemble(L)
    end_assemble = time.time()
    Time_assemble_standard_ex.append(end_assemble-start_assemble)
    bc.apply(A,B) # apply Dirichlet boundary conditions to the problem
    start_standard = time.time()
    u = df.Function(V)
    df.solve(A,u.vector(),B)
    end_standard = time.time()
    Time_solve_standard_ex.append(end_standard-start_standard)
    Time_total_standard_ex.append(Time_assemble_standard_ex[-1] + Time_solve_standard_ex[-1])
    # Compute and store h and L2 H1 errors
    hh_standard_ex.append(mesh.hmax())
    error_l2_standard_ex.append((df.assemble((((u_ex-u))**2)*df.dx)**(0.5))/(df.assemble((((u_ex))**2)*df.dx)**(0.5)))            
    error_h1_standard_ex.append((df.assemble(((df.grad(u_ex-u))**2)*df.dx)**(0.5))/(df.assemble(((df.grad(u_ex))**2)*df.dx)**(0.5)))

# we compute the phi-fem for different sizes of cells
for i in range(start,end,step): 
    print("Phi-fem iteration : ", i)
    # we define parameters and the "global" domain O
    H = 10*2**i
    circle_dom = mshr.Circle(df.Point(0.5,0.5), df.sqrt(2.0)/4.0)
    init_mesh = mshr.generate_mesh(circle_dom, H)
    boundary_mesh = df.BoundaryMesh(init_mesh, "exterior", True)
    points = boundary_mesh.coordinates()
    boundary_mesh_tree = boundary_mesh.bounding_box_tree()
    tree = init_mesh.bounding_box_tree()
    nbr_points.append(len(points))
    plot(init_mesh, N=3, at = 0, lw=0,interactive = False, text ="Maillage initial \n " + str(init_mesh.num_cells()) + " cellules \n de taille h = " + str(init_mesh.hmax()))

    nbr_cells = df.sqrt(2.0)/hh_standard_ex[i-1]
    square = df.UnitSquareMesh(int(nbr_cells),int(nbr_cells)) # modifier H pour avoir hmax = hmax du cercle
    # We now define Omega using phi
    V_phi = df.FunctionSpace(square, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    plot(phi, at= 1, lw =0, interactive=False, text= r'Valeurs de phi')
    Cell_omega = df.MeshFunction("size_t", square, 2)
    Cell_omega.set_all(0)
    for cell in df.cells(square):  
        v1,v2,v3 = df.vertices(cell)
        if(phi(v1.point()) <= 0.0 or phi(v2.point()) <= 0.0 or phi(v3.point()) <= 0.0 or df.near(phi(v1.point()),0.0) or df.near(phi(v2.point()),0.0) or df.near(phi(v3.point()),0.0)):
            Cell_omega[cell] = 1
    mesh = df.SubMesh(square, Cell_omega, 1) 
    hh_phi.append(mesh.hmax()) # store the size of each element for this iteration  
    plot(mesh, at=2, lw=0,interactive = False, text="Reconstruction du domaine  \n " + str(mesh.num_cells()) + " cellules " +"\n" +"de taille h = " + str(mesh.hmax()))
    #interactive()
    screenshot('creating_phi/circle'+str(i)+'.png')
    closePlotter()    
    # Creation of the FunctionSpace for Phi on the new mesh
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    
    # Selection of cells and facets on the boundary
    mesh.init(1,2) 
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1) # codimension 1
    Cell = df.MeshFunction("size_t", mesh, 2)  # codimension 0
    Facet.set_all(0)
    Cell.set_all(0)
    for cell in df.cells(mesh) : 
        for facet in df.facets(cell): 
            v1,v2 = df.vertices(facet) 
            if(phi(v1.point())*phi(v2.point()) <= 0.0 or df.near(phi(v1.point())*phi(v2.point()),0.0)) : # si on est sur le bord 
                Cell[cell] = 1  
                for facett in df.facets(cell):
                    Facet[facett] = 1 
    
    # Variationnal problem resolution
    V = df.FunctionSpace(mesh, 'CG', degV)
    w_h = df.TrialFunction(V)
    v_h = df.TestFunction(V)
    n = df.FacetNormal(mesh)
    h = df.CellDiameter(mesh)
    h_avg = (h('+') + h('-'))/2.0
    # Exact solution and computation of the right hand side term f
    u_ex = df.Expression("exp(x[0])*sin(2*pi*x[1])", degree=4, domain=mesh) # solution exacte 
    f = -df.div(df.grad(u_ex))
    u_D = df.Expression("exp(x[0])*sin(2*pi*x[1])",degree=4,domain=mesh)
    u_D = u_D * (1 + phi)
    # Modification of dolfin measures, to consider cells and facets 
    dx = df.Measure("dx", mesh, subdomain_data = Cell)
    ds = df.Measure("ds", mesh)
    dS = df.Measure("dS", mesh, subdomain_data = Facet)
    
    sigma = 20 # Stabilization parameter
    # Creation of the bilinear and linear forms using stabilization terms and boundary condition
    a = df.inner(df.grad(phi*w_h),df.grad(phi*v_h))*dx  - df.inner(df.grad(phi*w_h),n)*phi*v_h*ds \
        + sigma*h_avg*df.jump(df.grad(phi*w_h),n)*df.jump(df.grad(phi*v_h),n)*dS(1) \
        + sigma*h**2*df.div(df.grad(phi*w_h))*df.div(df.grad(phi*v_h))*dx(1)  
    L = f*phi*v_h*dx  - sigma*h**2*f*df.div(df.grad(phi*v_h))*dx(1) \
        - sigma*h**2*df.div(df.grad(phi*v_h))*df.div(df.grad(u_D))*dx(1) \
        - df.inner(df.grad(u_D),df.grad(phi*v_h))*dx + df.inner(df.grad(u_D),n)*phi*v_h*ds \
        - sigma*h_avg*df.jump(df.grad(u_D),n)*df.jump(df.grad(phi*v_h),n)*dS(1) 
    
    start_assemble = time.time()
    A = df.assemble(a)
    B = df.assemble(L)
    end_assemble = time.time()
    Time_assemble_phi.append(end_assemble-start_assemble)
    w_h = df.Function(V)
    start_solve = time.time()
    df.solve(A, w_h.vector(),B)
    end_solve = time.time()
    Time_solve_phi.append(end_solve-start_solve)
    Time_total_phi.append(Time_assemble_phi[-1] + Time_solve_phi[-1])
    u_h = phi*w_h + u_D 
    # Compute and store relative error for H1 and L2 norms
    error_l2_phi.append((df.assemble((((u_ex-u_h))**2)*dx)**(0.5))/(df.assemble((((u_ex))**2)*dx)**(0.5)))            
    error_h1_phi.append((df.assemble(((df.grad(u_ex-u_h))**2)*dx)**(0.5))/(df.assemble(((df.grad(u_ex))**2)*dx)**(0.5)))

# Plot results 
plt.figure()
plt.loglog(hh_phi,error_l2_phi,'-o', label=r'$\phi$-fem $L^2$ error')
plt.loglog(hh_phi,error_h1_phi,'-o', label=r'$\phi$-fem $H^1$ error')
plt.loglog(hh_standard_ex,error_l2_standard_ex, '->',label=r'Standard fem $L^2$ error')
plt.loglog(hh_standard_ex,error_h1_standard_ex, '->',label=r'Standard fem $H^1$ error')
plt.loglog(hh_phi,[hhh**2 for hhh in hh_phi], '.',label="Quadratic")
plt.loglog(hh_phi, hh_phi, '.', label="Linear")
plt.xlabel("h")
plt.ylabel("error")
plt.legend()
plt.title(r'Relative error : $ \|\| u-u_h \|\| / \|\|u\|\| $ for $L^2$ and $H^1$ norms')
plt.savefig('relative_error_P_{name0}.png'.format(name0=degV))
plt.show()
plt.figure()
plt.semilogy(nbr_points,error_l2_phi,'-o', label=r'$\phi$-fem $L^2$ error')
plt.semilogy(nbr_points,error_h1_phi,'-o', label=r'$\phi$-fem $H^1$ error')
plt.semilogy(nbr_points,error_l2_standard_ex, '->',label=r'Standard fem $L^2$ error')
plt.semilogy(nbr_points,error_h1_standard_ex, '->',label=r'Standard fem $H^1$ error')
plt.xlabel("Number of points")
plt.ylabel("error")
plt.legend()
plt.title(r'Relative error : $ \|\| u-u_h \|\| / \|\|u\|\| $ for $L^2$ and $H^1$ norms')
plt.savefig('relative_error_P_{name0}_points.png'.format(name0=degV))
plt.show()
