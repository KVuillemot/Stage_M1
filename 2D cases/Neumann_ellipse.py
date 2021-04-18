import dolfin as df 
import matplotlib.pyplot as plt 
import multiphenics as mph 
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
        value[0] = -1.0 + (x[0]-0.5)**2/(df.sqrt(2.0)/4.0)**2 + (x[1]-0.5)**2/(1.0/16.0) 
#        value[0] = -1.0/8.0 + (x[0]-0.5)**2 + (x[1]-0.5)**2 

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
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1) 
    Cell = df.MeshFunction("size_t", mesh, 2)  
    cell_sub = df.MeshFunction("bool", mesh, 2)
    facet_sub = df.MeshFunction("bool", mesh, 1)
    vertices_sub = df.MeshFunction("bool", mesh, 0)
    Facet.set_all(0)
    Cell.set_all(0)
    cell_sub.set_all(0)
    facet_sub.set_all(0)
    vertices_sub.set_all(0)

    for cell in df.cells(mesh) :
        for facet in df.facets(cell): 
            v1,v2 = df.vertices(facet) 
            if(phi(v1.point())*phi(v2.point()) <= 0.0 or df.near(phi(v1.point())*phi(v2.point()),0.0)) : 
                Cell[cell] = 1  
                cell_sub[cell] = 1
                for facett in df.facets(cell):  
                    Facet[facett] = 1  
                    facet_sub[facett] = 1
                    v1, v2 = df.vertices(facett)
                    vertices_sub[v1], vertices_sub[v2] = 1,1
    File2 = df.File("sub.rtc.xml/mesh_function_2.xml")
    File2 << cell_sub
    File1 = df.File("sub.rtc.xml/mesh_function_1.xml")
    File1 << facet_sub
    File0 = df.File("sub.rtc.xml/mesh_function_0.xml")
    File0 << vertices_sub
    
    # Beginning of variationnal problem resolution
    yp_res = mph.MeshRestriction(mesh,"sub.rtc.xml")
    V = df.FunctionSpace(mesh, "CG", degV)
    Z = df.VectorFunctionSpace(mesh,"CG",degV, dim = 2)
    Q = df.FunctionSpace(mesh,"DG",degV-1)
    W = mph.BlockFunctionSpace([V,Z,Q], restrict=[None,yp_res,yp_res])
    uyp = mph.BlockTrialFunction(W)
    (u, y, p) = mph.block_split(uyp)
    vzq = mph.BlockTestFunction(W)
    (v, z, q) = mph.block_split(vzq)

    dx = df.Measure("dx", mesh, subdomain_data = Cell)
    ds = df.Measure("ds", mesh)
    dS = df.Measure("dS", mesh, subdomain_data = Facet)

    gamma_div, gamma_u, gamma_p, sigma = 1.0, 1.0, 1.0, 0.01
    h = df.CellDiameter(mesh)
    n = df.FacetNormal(mesh)
    u_ex = df.Expression("sin(x[0])*exp(x[1])", degree = 6, domain = mesh)
    f = -df.div(df.grad(u_ex)) + u_ex
    g = df.inner(df.grad(u_ex),df.grad(phi))/(df.inner(df.grad(phi),df.grad(phi))**0.5) + u_ex*phi
    
    # Construction of the bilinear and linear forms
    auv = df.inner(df.grad(u),df.grad(v))*dx + u*v*dx + gamma_div*u*v*dx(1) + gamma_u*df.inner(df.grad(u),df.grad(v))*dx(1) \
        + sigma*df.avg(h)*df.inner(df.jump(df.grad(u),n),df.jump(df.grad(v),n))*dS(1) 
    ayv = df.inner(y,n*v)*ds + gamma_div*df.div(y)*v*dx(1) + gamma_u*df.inner(y,df.grad(v))*dx(1)
    apv = 0.0
    auz = gamma_div*u*df.div(z)*dx(1) + gamma_u*df.inner(df.grad(u),z)*dx(1)
    ayz = gamma_div*df.div(y)*df.div(z)*dx(1) + gamma_u*df.inner(y,z)*dx(1) \
        + (gamma_p/(h**2))*(df.inner(y,df.grad(phi)))*df.inner(z,df.grad(phi))*dx(1)
    apz = (gamma_p/(h**3))*p*phi*(df.inner(z,df.grad(phi)))*dx(1)
    auq = 0.0 
    ayq = (gamma_p/(h**3))*df.inner(y,df.grad(phi))*q*phi*dx(1)
    apq = (gamma_p/(h**4))*p*phi*q*phi*dx(1)
    a = [[auv, auz, auq],
            [ayv, ayz, ayq],
            [apv, apz, apq]]
    lv = f*v*dx + gamma_div*f*v*dx(1)
    lz = gamma_div*f*df.div(z)*dx(1) - (gamma_p/(h**2))*g*df.inner(df.grad(phi),df.grad(phi))**(0.5)*df.inner(z,df.grad(phi))*dx(1)
    lq = - (gamma_p/(h**3))*g*df.inner(df.grad(phi),df.grad(phi))**(0.5)*q*phi*dx(1)
    l = [lv, lz, lq]
    start_assemble = time.time()
    A = mph.block_assemble(a)
    B = mph.block_assemble(l)
    end_assemble = time.time()
    Time_assemble_phi.append(end_assemble-start_assemble)
    UU = mph.BlockFunction(W)
    start_solve = time.time()
    mph.block_solve(A, UU.block_vector(), B)
    end_solve = time.time()
    Time_solve_phi.append(end_solve-start_solve)
    Time_total_phi.append(Time_assemble_phi[-1] + Time_solve_phi[-1])
    u_h = UU[0]
    # Compute and store relative error for H1 and L2 norms
    error_l2_phi.append((df.assemble((((u_ex-u_h))**2)*dx)**(0.5))/(df.assemble((((u_ex))**2)*dx)**(0.5)))            
    error_h1_phi.append((df.assemble(((df.grad(u_ex-u_h))**2)*dx)**(0.5))/(df.assemble(((df.grad(u_ex))**2)*dx)**(0.5)))


# Computation of the standard FEM       
domain = mshr.Ellipse(df.Point(0.5,0.5),df.sqrt(2.0)/4.0, 1.0/4.0) # create of the domain
for i in range(start, end, step):
    H = 25*2**(i-1) # to have approximately the same precision as in the phi-fem computation
    mesh = mshr.generate_mesh(domain,H)
    print("Standard fem iteration : ", i)
    # FunctionSpace P1
    V = df.FunctionSpace(mesh, 'CG', degV)  
    v = df.TestFunction(V)
    u = df.TrialFunction(V)
    u_ex = df.Expression("sin(x[0])*exp(x[1])", degree = 6, domain = mesh)
    f = - df.div(df.grad(u_ex)) + u_ex 
    phi = df.Expression("-1.0 + pow(x[0]-0.5,2)/pow(sqrt(2.0)/4.0,2) + pow(x[1]-0.5,2)/pow(1.0/4.0,2)", degree = 6, domain = mesh)
    g = df.inner(df.grad(u_ex),df.grad(phi))/df.inner(df.grad(phi),df.grad(phi))**(0.5) + u_ex*phi
    # Resolution of the variationnal problem
    a = df.inner(df.grad(u), df.grad(v))*df.dx + u*v*df.dx 
    l = f*v*df.dx + g*v*df.ds
    start_assemble = time.time()
    A = df.assemble(a)
    B = df.assemble(l)
    end_assemble = time.time()
    Time_assemble_standard.append(end_assemble-start_assemble)
    start_standard = time.time()
    u = df.Function(V)
    df.solve(A,u.vector(),B)
    end_standard = time.time()
    u_h = u
    Time_solve_standard.append(end_standard-start_standard)
    Time_total_standard.append(Time_assemble_standard[-1] + Time_solve_standard[-1])
    # Compute and store h and L2 H1 errors
    hh_standard.append(mesh.hmax())
    error_l2_standard.append((df.assemble((((u_ex-u_h))**2)*df.dx)**(0.5))/(df.assemble((((u_ex))**2)*df.dx)**(0.5)))            
    error_h1_standard.append((df.assemble(((df.grad(u_ex-u_h))**2)*df.dx)**(0.5))/(df.assemble(((df.grad(u_ex))**2)*df.dx)**(0.5)))

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
f = open('output_ghost_case1_Neumann2_P{name0}.txt'.format(name0=degV),'w')
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
