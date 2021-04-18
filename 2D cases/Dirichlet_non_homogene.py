import dolfin as df 
import matplotlib.pyplot as plt 
import mshr
import time

# dolfin parameters
df.parameters["ghost_mode"] = "shared_facet" 
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters['allow_extrapolation'] = True
df.parameters["form_compiler"]["representation"] = 'uflacs'

# degree of interpolation for V and Vphi
degV = 1
degPhi = 1 + degV

# Function used to write in the outputs files
def output_latex(f,A,B):
	for i in range(len(A)):
		f.write('(')
		f.write(str(A[i]))
		f.write(',')
		f.write(str(B[i]))
		f.write(')\n')
	f.write('\n')
"""
We define the level-sets function phi :
Here we consider the case of circle centered in (0.5,0.5) of radius sqrt(2)/4, so the level-sets is the equation of the circle.
"""
class phi_expr(df.UserExpression) : 
    def eval(self, value, x):
        value[0] = -1.0/8.0 + (x[0]-0.5)**2 + (x[1]-0.5)**2 
    def value_shape(self):
        return (2,)

# We create the lists that we'll use to store errors and computation time for the phi-fem and standard fem
Time_assemble_phi, Time_solve_phi, Time_total_phi, error_l2_phi, error_h1_phi, hh_phi = [], [], [], [], [], []
Time_assemble_standard, Time_solve_standard, Time_total_standard, error_h1_standard, error_l2_standard,  hh_standard = [], [], [], [], [], []

# we compute the phi-fem for different sizes of cells
start,end,step = 1,5,1
for i in range(start,end,step): 
    print("Phi-fem iteration : ", i)
    # we define parameters and the "global" domain O
    H = 25*2**i
    square = df.UnitSquareMesh(H,H)
    
    # We now define Omega using phi
    V_phi = df.FunctionSpace(square, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    Cell_omega = df.MeshFunction("size_t", square, 2)
    Cell_omega.set_all(0)
    for cell in df.cells(square):  
        v1,v2,v3 = df.vertices(cell)
        if(phi(v1.point()) <= 0.0 or phi(v2.point()) <= 0.0 or phi(v3.point()) <= 0.0 or df.near(phi(v1.point()),0.0) or df.near(phi(v2.point()),0.0) or df.near(phi(v3.point()),0.0)):
            Cell_omega[cell] = 1
    mesh = df.SubMesh(square, Cell_omega, 1) 
    hh_phi.append(mesh.hmax()) # store the size of each element for this iteration  
    
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
    u_D = df.Expression("exp(x[0])*sin(2*pi*x[1])*(1+(-1.0/8.0 + (x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5)))",degree=4,domain=mesh)

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

# Computation of the standard FEM       
domain = mshr.Circle(df.Point(0.5,0.5),df.sqrt(2.0)/4.0) # creation of the domain
for i in range(start, end, step):
    H = 25*2**(i-1) # to have approximately the same precision as in the phi-fem computation
    mesh = mshr.generate_mesh(domain,H)
    print("Standard fem iteration : ", i)
    # FunctionSpace P1
    V = df.FunctionSpace(mesh, 'CG', degV)  
    v = df.TestFunction(V)
    u = df.TrialFunction(V)
    u_ex = df.Expression("exp(x[0])*sin(2*pi*x[1])",degree=4,domain=mesh)
    u_D = df.Expression("exp(x[0])*sin(2*pi*x[1])*(1+(-1.0/8.0 + (x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5)))",degree=4,domain=mesh)
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
    Time_assemble_standard.append(end_assemble-start_assemble)
    bc.apply(A,B) # apply Dirichlet boundary conditions to the problem
    start_standard = time.time()
    u = df.Function(V)
    df.solve(A,u.vector(),B)
    end_standard = time.time()
    Time_solve_standard.append(end_standard-start_standard)
    Time_total_standard.append(Time_assemble_standard[-1] + Time_solve_standard[-1])
    # Compute and store h and L2 H1 errors
    hh_standard.append(mesh.hmax())
    error_l2_standard.append((df.assemble((((u_ex-u))**2)*df.dx)**(0.5))/(df.assemble((((u_ex))**2)*df.dx)**(0.5)))            
    error_h1_standard.append((df.assemble(((df.grad(u_ex-u))**2)*df.dx)**(0.5))/(df.assemble(((df.grad(u_ex))**2)*df.dx)**(0.5)))

# Plot results : error/precision, Time/precision, Time/error and Total_time/error
plt.figure()
plt.loglog(hh_phi,error_l2_phi,'-o', label=r'$\phi$-fem $L^2$ error')
plt.loglog(hh_phi,error_h1_phi,'-o', label=r'$\phi$-fem $H^1$ error')
plt.loglog(hh_standard,error_l2_standard, '-x',label=r'Standard fem $L^2$ error')
plt.loglog(hh_standard,error_h1_standard, '-x',label=r'Standard fem $H^1$ error')
plt.loglog(hh_phi,[hhh**2 for hhh in hh_phi], '.',label="Quadratic")
plt.loglog(hh_phi, hh_phi, '.', label="Linear")
plt.xlabel("h")
plt.ylabel("error")
plt.legend()
plt.title(r'Relative error : $ \|\| u-u_h \|\| / \|\|u\|\| $ for $L^2$ and $H^1$ norms')
plt.savefig('relative_error_P_{name0}.png'.format(name0=degV))
plt.show()
plt.figure()
plt.loglog(hh_phi,Time_assemble_phi, '-o',label=r'assemble $\phi$-FEM')
plt.loglog(hh_phi,Time_solve_phi,'-o', label=r'solve $\phi$-FEM')
plt.loglog(hh_standard,Time_assemble_standard, '-x',label=r' assemble standard FEM')
plt.loglog(hh_standard,Time_solve_standard,'-x', label=r'solve standard FEM')
plt.xlabel("h")
plt.ylabel("Time (s)")
plt.legend()
plt.title("Computing time")
plt.savefig('Time_precision_P_{name0}.png'.format(name0=degV))
plt.show()
plt.figure()
plt.loglog(error_l2_phi,Time_assemble_phi, '-o',label=r' Assemble $\phi$-fem')
plt.loglog(error_l2_phi,Time_solve_phi,'-o', label=r' Solve $\phi$-fem')
plt.loglog(error_l2_standard,Time_assemble_standard,'-x', label="Assemble standard FEM")
plt.loglog(error_l2_standard,Time_solve_standard,'-x', label="Solve standard FEM")
plt.xlabel(r'$\| \|u-u_h \|\|_{L^2}$')
plt.ylabel("Time (s)")
plt.title(r'Computing time')
plt.legend()
plt.savefig('Time_error_P_{name0}.png'.format(name0=degV))
plt.show()
plt.figure()
plt.loglog(error_l2_phi,Time_total_phi,'-o', label=r'$\phi$-fem')
plt.loglog(error_l2_standard,Time_total_standard,'-x', label="Standard FEM")
plt.xlabel(r'$\| \|u-u_h \|\|_{L^2}$')
plt.ylabel("Time (s)")
plt.title(r'Computing time')
plt.legend()
plt.savefig('Total_time_error_P_{name0}.png'.format(name0=degV))
plt.show()

#  Write the output file for latex
f = open('output_ghost_case1_dirichlet_P{name0}.txt'.format(name0=degV),'w')
f.write('relative L2 norm phi fem: \n')	
output_latex(f, hh_phi, error_l2_phi)
f.write('relative H1 norm phi fem : \n')	
output_latex(f, hh_phi, error_h1_phi)
f.write('relative L2 norm and time phi fem : \n')	
output_latex(f, error_l2_phi, Time_total_phi)
f.write('relative H1 norm and time phi fem : \n')	
output_latex(f, error_h1_phi, Time_total_phi)
f.write('relative L2 norm classic fem: \n')	
output_latex(f, hh_standard, error_l2_standard)
f.write('relative H1 normclassic fem : \n')	
output_latex(f, hh_standard, error_h1_standard)
f.write('relative L2 norm and time classic fem : \n')	
output_latex(f, error_l2_standard, Time_total_standard)
f.write('relative H1 norm and time classic fem : \n')	
output_latex(f, error_h1_standard, Time_total_standard)
f.close()
