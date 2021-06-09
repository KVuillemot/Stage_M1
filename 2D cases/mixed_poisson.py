import dolfin as df 
import matplotlib.pyplot as plt 
import multiphenics as mph 
import mshr
import time
from vedo.dolfin import plot, interactive
from matplotlib import rc, rcParams 

plt.style.use('bmh') 
params = {'axes.labelsize': 28,
          'font.size': 22,
          'axes.titlesize': 28,
          'legend.fontsize': 20,
          'figure.titlesize': 26,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'text.usetex': True,
          'figure.figsize':(10,8),          
          'legend.shadow': True,
          'patch.edgecolor': 'black'}
plt.rcParams.update(params)
# dolfin parameters
df.parameters["ghost_mode"] = "shared_facet" 
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters['allow_extrapolation'] = True
df.parameters["form_compiler"]["representation"] = 'uflacs'

# degree of interpolation for V and Vphi
degV = 2
degPhi = 2 + degV

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
    
def dirichlet(point):
    x,y = point.x(), point.y()
    return x >= 0.5
def neumann(point):
    return not(dirichlet(point))
# We create the lists that we'll use to store errors and computation time for the phi-fem and standard fem
Time_assemble_phi, Time_solve_phi, Time_total_phi, error_l2_phi, error_h1_phi, hh_phi = [], [], [], [], [], []
Time_assemble_standard, Time_solve_standard, Time_total_standard, error_h1_standard, error_l2_standard,  hh_standard = [], [], [], [], [], []

# we compute the phi-fem for different sizes of cells
start,end,step = 1,5,1
for i in range(start,end,step): 
    print("Phi-fem iteration : ", i)
    # we define parameters and the "global" domain O
    H = 8*2**i
    square = df.UnitSquareMesh(H,H)
    
    # We now define Omega using phi
    V_phi = df.FunctionSpace(square, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    Cell_omega = df.MeshFunction("size_t", square, square.topology().dim())
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
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1) 
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())  
    cell_sub = df.MeshFunction("bool", mesh, mesh.topology().dim())
    facet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 1)
    vertices_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 2)
    Facet.set_all(0)
    Cell.set_all(0)
    cell_sub.set_all(0)
    facet_sub.set_all(0)
    vertices_sub.set_all(0)

    for cell in df.cells(mesh) :
        for facet in df.facets(cell): 
            v1,v2 = df.vertices(facet) 
            if(phi(v1.point())*phi(v2.point()) <= 0.0 or df.near(phi(v1.point())*phi(v2.point()),0.0)) : 
                # check if the cell is a cell for Dirichlet condition or Neumann condition and add every cells, facets, vertices to the restricition
                
                # Cells for dirichlet condition
                if dirichlet(v1.point()) and dirichlet(v2.point()): 
                    Cell[cell] = 2
                    cell_sub[cell] = 1
                    for facett in df.facets(cell):  
                        Facet[facett] = 2
                        facet_sub[facett] = 1
                        v1, v2 = df.vertices(facett)
                        vertices_sub[v1], vertices_sub[v2] = 1,1
                # Cells for Neumann condition
                else : 
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
    V = df.FunctionSpace(mesh, 'CG', degV)
    u_ex = df.Expression(('sin(x[0]) * exp(x[1])'), degree = 6, domain = mesh)
    f = - df.div(df.grad(u_ex))   
    Z = df.VectorFunctionSpace(mesh,"CG",degV, dim =2)
    Q = df.FunctionSpace(mesh,"DG",degV-1)
    W = mph.BlockFunctionSpace([V,Z,Q], restrict=[None,yp_res,yp_res])
    uyp = mph.BlockTrialFunction(W)
    (u, y, p) = mph.block_split(uyp)
    vzq = mph.BlockTestFunction(W)
    (v, z, q) = mph.block_split(vzq)

    dx = df.Measure("dx", mesh, subdomain_data = Cell)
    ds = df.Measure("ds", mesh, subdomain_data = Facet) # considering facets to apply Dirichlet or Neumann for the boundary condition
    dS = df.Measure("dS", mesh, subdomain_data = Facet)

    gamma_div, gamma_u, gamma_p, sigma_p, gamma_D, sigma_D = 1.0, 1.0, 1.0, 0.01, 20.0, 20.0
    h = df.CellDiameter(mesh)
    n = df.FacetNormal(mesh)
    g = df.dot(df.grad(u_ex),df.grad(phi))/(df.inner(df.grad(phi),df.grad(phi))**0.5) \
        + u_ex*phi
    u_D = u_ex * (1 + phi)
    # Construction of the bilinear and linear forms
    boundary_penalty = sigma_p*df.avg(h)*df.inner(df.jump(df.grad(u),n), df.jump(df.grad(v),n))*dS(1) \
                     + sigma_D*df.avg(h)*df.inner(df.jump(df.grad(u),n), df.jump(df.grad(v),n))*dS(2) \
                     + sigma_D*h**2*(df.inner(- df.div(df.grad(u)) ,- df.div(df.grad(v)) ))*dx(2)
    
    phi_abs = df.inner(df.grad(phi),df.grad(phi))**0.5

    auv = df.inner(df.grad(u), df.grad(v))*dx  \
        + gamma_u*df.inner(df.grad(u),df.grad(v))*dx(1) \
        + boundary_penalty \
        + gamma_D*h**(-2)*df.inner(u,v)*dx(2) \
        - df.inner(df.dot(df.grad(u),n),v)*ds(2) 

    auz = gamma_u*df.inner(df.grad(u),z)*dx(1) 
    auq = - gamma_D*h**(-3)*df.dot(u,q*phi)*dx(2)  
    
    ayv = df.inner(df.dot(y,n),v)*ds(1) + gamma_u*df.inner(y,df.grad(v))*dx(1) 
        
    ayz = gamma_u*df.inner(y,z)*dx(1) + gamma_div*df.inner(df.div(y), df.div(z))*dx(1) \
        + gamma_p*h**(-2)*df.inner(df.dot(y,df.grad(phi)), df.dot(z,df.grad(phi)))*dx(1)
    ayq = gamma_p*h**(-3)*df.inner(df.dot(y,df.grad(phi)), q*phi)*dx(1)
    
    apv = - gamma_D*h**(-3)*df.dot(v,p*phi)*dx(2) 
    apz = gamma_p*h**(-3)*df.inner(p*phi, df.dot(z,df.grad(phi)))*dx(1)
    apq = gamma_p*h**(-4)*df.inner(p*phi,q*phi)*dx(1) \
        + gamma_D*h**(-4)*df.inner(p*phi,q*phi)*dx(2)
    
    lv = df.inner(f,v)*dx  \
        + sigma_D*h**2*df.inner(f, - df.div(df.grad(v)))*dx(2) \
        + gamma_D*h**(-2)*df.dot(u_D,v)*dx(2)
    lz = df.inner(f, df.div(z))*dx(1) - gamma_p*h**(-2)*df.inner(g*phi_abs, df.dot(z,df.grad(phi)))*dx(1)
    lq = - gamma_p*h**(-3)*df.inner(g*phi_abs,q*phi)*dx(1) - gamma_D*h**(-3)*df.inner(u_D,q*phi)*dx(2)
    
    a = [[auv,auz,auq],
         [ayv,ayz,ayq],
         [apv,apz,apq]]
    l = [lv,lz,lq]  
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
    u_h = df.interpolate(UU[0],V)
    if i == end  :
        
        plot(mesh, u_h, N = 3, interactive = False, at = 0, text = r'$\phi$-fem')
    # Compute and store relative error for H1 and L2 norms
    error_l2_phi.append((df.assemble((((u_ex-u_h))**2)*dx)**(0.5))/(df.assemble((((u_ex))**2)*dx)**(0.5)))            
    error_h1_phi.append((df.assemble(((df.grad(u_ex-u_h))**2)*dx)**(0.5))/(df.assemble(((df.grad(u_ex))**2)*dx)**(0.5)))

# Computation of the standard FEM       
domain = mshr.Circle(df.Point(0.5,0.5),df.sqrt(2.0)/4.0) # create of the domain
for i in range(start, end, step):
    H = 8*2**(i-1) # to have approximately the same precision as in the phi-fem computation
    mesh = mshr.generate_mesh(domain,H)
    print("Standard fem iteration : ", i)
    # FunctionSpace P1
    u_ex = df.Expression(('sin(x[0]) * exp(x[1])'), degree = 6, domain = mesh)
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    f = - df.div(df.grad(u_ex)) 
    V = df.FunctionSpace(mesh, 'CG', degV)
    boundary = 'on_boundary && x[0] >= 0.5'
    u_D = u_ex * ( 1 + phi)
    bc = df.DirichletBC(V, u_D, boundary)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    n = df.FacetNormal(mesh)
    g = df.dot(df.grad(u_ex),df.grad(phi))/(df.inner(df.grad(phi),df.grad(phi))**0.5) \
        + u_ex*phi
    a = df.inner(df.grad(u), df.grad(v))*df.dx 
    L = df.dot(f,v)*df.dx + df.dot(g,v)*df.ds
    start_assemble = time.time()
    A = df.assemble(a)
    B = df.assemble(L)
    end_assemble = time.time()
    Time_assemble_standard.append(end_assemble-start_assemble)
    u = df.Function(V)
    bc.apply(A,B) # apply Dirichlet boundary conditions to the problem    
    start_solve = time.time()
    df.solve(A, u.vector(), B)
    end_solve = time.time()
    Time_solve_standard.append(end_solve-start_solve)
    Time_total_standard.append(Time_assemble_standard[-1] + Time_solve_standard[-1])
    #plot(u, at = 1, warpZfactor = 4, interactive=False, text = r'standard fem with ' + '\n' + 'h = ' + str(mesh.hmax())) 
    u_h = df.interpolate(u, V)
    if i == end :
        plot(u_h, at = 1, text = 'standard fem')
        plot(mesh, u_ex, at = 2, text = 'exact solution')
    # Compute and store h and L2 H1 errors
    hh_standard.append(mesh.hmax())
    error_l2_standard.append((df.assemble((((u_ex-u_h))**2)*df.dx)**(0.5))/(df.assemble((((u_ex))**2)*df.dx)**(0.5)))            
    error_h1_standard.append((df.assemble(((df.grad(u_ex-u_h))**2)*df.dx)**(0.5))/(df.assemble(((df.grad(u_ex))**2)*df.dx)**(0.5)))

# Plot results : error/precision, Time/precision, Time/error and Total_time/error
plt.figure()
plt.loglog(hh_phi,error_h1_phi,'o--', label=r'$\phi$-FEM $H^1$')
plt.loglog(hh_phi,error_l2_phi,'o-', label=r'$\phi$-FEM $L^2$')
plt.loglog(hh_standard,error_h1_standard, '--x',label=r'Std FEM $H^1$')
plt.loglog(hh_standard,error_l2_standard, '-x',label=r'Std FEM $L^2$')
if degV == 1 :
    plt.loglog(hh_phi, hh_phi, '.', label="Linear")
    plt.loglog(hh_phi,[hhh**2 for hhh in hh_phi], '.',label="Quadratic")
elif degV == 2 :
    plt.loglog(hh_phi,[hhh**2 for hhh in hh_phi], '.',label="Quadratic")
    plt.loglog(hh_phi,[hhh**3 for hhh in hh_phi], '.',label="Cubic")

plt.xlabel("$h$")
plt.ylabel(r'$\frac{\|u-u_h\|}{\|u\|}$')
plt.legend(loc='upper right', ncol=2)
plt.title(r'Relative error : $ \frac{\|u-u_h\|}{\|u\|} $ for $L^2$ and $H^1$ norms', y=1.025)
plt.tight_layout()
plt.savefig('mixed_poisson/relative_error_P_{name0}.png'.format(name0=degV))
plt.show()

plt.figure()
plt.loglog(hh_phi,Time_assemble_phi, '-o',label=r'Assemble $\phi$-FEM')
plt.loglog(hh_phi,Time_solve_phi,'--o', label=r'Solve $\phi$-FEM')
plt.loglog(hh_standard,Time_assemble_standard, '-x',label=r'Assemble standard FEM')
plt.loglog(hh_standard,Time_solve_standard,'--x', label=r'Solve standard FEM')
plt.xlabel("$h$")
plt.ylabel("Time (s)")
plt.legend(loc='upper right')
plt.title("Computing time")
plt.tight_layout()
plt.savefig('mixed_poisson/Time_precision_P_{name0}.png'.format(name0=degV))
plt.show()
plt.figure()
plt.loglog(error_l2_phi,Time_assemble_phi, '-o',label=r'Assemble $\phi$-fem')
plt.loglog(error_l2_phi,Time_solve_phi,'--o', label=r'Solve $\phi$-fem')
plt.loglog(error_l2_standard,Time_assemble_standard,'-x', label="Assemble standard FEM")
plt.loglog(error_l2_standard,Time_solve_standard,'--x', label="Solve standard FEM")
plt.xlabel(r'$\frac{\|u-u_h\|_{L^2}}{\|u\|_{L^2}}$')
plt.ylabel("Time (s)")
plt.title(r'Computing time')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('mixed_poisson/Time_error_P_{name0}.png'.format(name0=degV))
plt.show()
plt.figure()
plt.loglog(error_l2_phi,Time_total_phi,'-o', label=r'$\phi$-fem')
plt.loglog(error_l2_standard,Time_total_standard,'-x', label="Standard FEM")
plt.xlabel(r'$\frac{\|u-u_h\|_{L^2}}{\|u\|_{L^2}}$')
plt.ylabel("Time (s)")
plt.title(r'Computing time')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('mixed_poisson/Total_time_error_P_{name0}.png'.format(name0=degV))
plt.show()
#  Write the output file for latex
f = open('mixed_poisson/output_ghost_case1_Neumann_P{name0}.txt'.format(name0=degV),'w')
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
