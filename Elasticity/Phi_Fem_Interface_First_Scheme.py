import dolfin as df 
import matplotlib.pyplot as plt 
import mshr
from matplotlib import rc, rcParams
import multiphenics as mph

# plot parameters
plt.style.use('bmh') 
params = {'axes.labelsize': 'large',
          'font.size': 22,
          'axes.titlesize': 'large',
          'legend.fontsize': 18,
          'figure.titlesize': 24,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
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

# Function used to write in the outputs files
def output_latex(f,A,B):
	for i in range(len(A)):
		f.write('(')
		f.write(str(A[i]))
		f.write(',')
		f.write(str(B[i]))
		f.write(')\n')
	f.write('\n')
 
# degree of interpolation for V and Vphi
degV = 2
degPhi = 2 + degV 

# expression of phi
class phi_expr(df.UserExpression) : 
    def eval(self, value, x):
        value[0] =  -0.3**2 + (x[0]-0.5)**2 + (x[1] - 0.5)**2 

    def value_shape(self):
        return (2,)

# functions and parameters for elasticity
def sigma1(u):
    return lambda_1 * df.div(u)*df.Identity(2) + 2.0*mu1*epsilon(u)

def sigma2(u):
    return lambda_2 * df.div(u)*df.Identity(2) + 2.0*mu2*epsilon(u)
def epsilon(u):
    return (1.0/2.0)*(df.grad(u) + df.grad(u).T)

E1 = 0.07
nu1 = 0.3
lambda_1 = E1*nu1/((1.0+nu1)*(1.0-2.0*nu1))
mu1 = E1/(2.0*(1.0+nu1))

E2 = 0.07
nu2 = 0.3
lambda_2 = E2*nu2/((1.0+nu2)*(1.0-2.0*nu2))
mu2 = E2/(2.0*(1.0+nu2))


error_h1, error_l2, hh = [], [],[]

start,end,step= 0,5,1

for i in range(start,end,step):
    H = 10*2**i
    # creation of the domain
    mesh = df.UnitSquareMesh(H,H)
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    
    hh.append(mesh.hmax())
    
    # initialization of mesh functions to create Omega1, Omega2 and the boundaries
    omega1, omega2, interf, gamma1, gamma2 = 1, 2, 3, 4, 5
    mesh.init(1,2) 
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1) 
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())  
    cell_sub = df.MeshFunction("bool", mesh, mesh.topology().dim())
    facet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim()-1)
    vertices_sub = df.MeshFunction("bool", mesh, 0)
    Facet.set_all(0)
    Cell.set_all(0)
    cell_sub.set_all(0)
    facet_sub.set_all(0)
    vertices_sub.set_all(0)
    
    # creation of Omega1 (including the interface)
    for cell in df.cells(mesh) :
        v1,v2,v3 = df.vertices(cell)
        if(phi(v1.point()) < 0.0 or phi(v2.point()) < 0.0 or phi(v3.point()) < 0.0 
        or df.near(phi(v1.point()),0.0) or df.near(phi(v2.point()),0.0) or df.near(phi(v3.point()),0.0)):
            Cell[cell] = omega1 
            cell_sub[cell] = 1
            for facett in df.facets(cell):  
                Facet[facett] = omega1
                facet_sub[facett] = 1
                v1, v2 = df.vertices(facett)
                vertices_sub[v1], vertices_sub[v2] = 1,1

    File2 = df.File("omega1.rtc.xml/mesh_function_2.xml")
    File2 << cell_sub
    File1 = df.File("omega1.rtc.xml/mesh_function_1.xml")
    File1 << facet_sub
    File0 = df.File("omega1.rtc.xml/mesh_function_0.xml")
    File0 << vertices_sub
    Omega1 = mph.MeshRestriction(mesh,"omega1.rtc.xml")
    
    # Creation of Omega2 (including the interface)
    cell_sub.set_all(0)
    facet_sub.set_all(0)
    vertices_sub.set_all(0)
    for cell in df.cells(mesh) :
        v1,v2,v3 = df.vertices(cell)
        if(phi(v1.point()) > 0.0 or phi(v2.point()) > 0.0 or phi(v3.point()) > 0.0 
            or df.near(phi(v1.point()),0.0) or df.near(phi(v2.point()),0.0) or df.near(phi(v3.point()),0.0)):
            Cell[cell] = omega2 
            cell_sub[cell] = 1
            for facett in df.facets(cell):
                Facet[facett] = omega2  
                facet_sub[facett] = 1
                v1, v2 = df.vertices(facett)
                vertices_sub[v1], vertices_sub[v2] = 1,1
    File2 = df.File("omega2.rtc.xml/mesh_function_2.xml")
    File2 << cell_sub
    File1 = df.File("omega2.rtc.xml/mesh_function_1.xml")
    File1 << facet_sub
    File0 = df.File("omega2.rtc.xml/mesh_function_0.xml")
    File0 << vertices_sub
    Omega2 = mph.MeshRestriction(mesh,"omega2.rtc.xml")

    # creation of the restricition for the interface
    cell_sub.set_all(0)
    facet_sub.set_all(0)
    vertices_sub.set_all(0)
    for cell in df.cells(mesh):  
        for facet in df.facets(cell): 
            v1,v2 = df.vertices(facet) 
            if(phi(v1.point())*phi(v2.point()) < 0.0 or df.near(phi(v1.point())*phi(v2.point()), 0.0)): 
                Cell[cell] = interf
                cell_sub[cell] = 1
                for facett in df.facets(cell):  
                    Facet[facett] = interf
                    facet_sub[facett] = 1
                    v1, v2 = df.vertices(facett)
                    vertices_sub[v1], vertices_sub[v2] = 1,1
                
    File2 = df.File("interface.rtc.xml/mesh_function_2.xml")
    File2 << cell_sub
    File1 = df.File("interface.rtc.xml/mesh_function_1.xml")
    File1 << facet_sub
    File0 = df.File("interface.rtc.xml/mesh_function_0.xml")
    File0 << vertices_sub
    Interface = mph.MeshRestriction(mesh,"interface.rtc.xml")
    
    for cell in df.cells(mesh):
        if Cell[cell] == omega2 :
            for facet in df.facets(cell) :
                if Facet[facet] == interf :
                    Facet[facet] = gamma1

        if Cell[cell] == omega1 :
            for facet in df.facets(cell) :
                if Facet[facet] == interf :
                    Facet[facet] = gamma2 
            
    # creation of the spaces 
    V = df.VectorFunctionSpace(mesh, 'CG', degV, dim=2)
    Z = df.TensorFunctionSpace(mesh,"CG",degV, shape = (2,2))
    Q = df.VectorFunctionSpace(mesh,"DG",degV-1, dim = 2)
    W = mph.BlockFunctionSpace([V,V,Z,Z,Q], restrict=[Omega1, Omega2, Interface, Interface, Interface])
    uyp = mph.BlockTrialFunction(W)
    (u1, u2, y1, y2, p) = mph.block_split(uyp)
    vzq = mph.BlockTestFunction(W)
    (v1, v2, z1, z2, q) = mph.block_split(vzq)
    
    # modification of the measures
    dx = df.Measure("dx", mesh, subdomain_data = Cell)
    ds = df.Measure("ds", mesh, subdomain_data = Facet)
    dS = df.Measure("dS", mesh, subdomain_data = Facet)
    
    # parameters for the considered case
    gamma_div, gamma_u, gamma_p, gamma_y, sigma_p = 10.0, 10.0, 10.0, 10.0, 1.0
    h = df.CellDiameter(mesh)
    n = df.FacetNormal(mesh)
            
    b = df.Expression('-pow(0.3,2) + pow(x[0]-0.5,2) + pow(x[1] - 0.5,2) < 0.0  ? k_1 : k_2', 
                        degree=4, k_1=1.0, k_2=0.0, domain = mesh)
    u_ex = df.Expression(('cos(x[0])*exp(x[1])', 'sin(x[1])*exp(x[0])'), domain = mesh, degree = 4)
    f = - df.div(b*sigma1(u_ex)+ (1.0-b)*sigma2(u_ex))
    u_D = u_ex 
    
    # Construction of the bilinear and linear forms
    Gh1 = sigma_p*df.avg(h)*df.inner(df.jump(sigma1(u1),n), df.jump(sigma1(v1),n))*(dS(interf)) 
    Gh2 = sigma_p*df.avg(h)*df.inner(df.jump(sigma2(u2),n), df.jump(sigma2(v2),n))*(dS(interf))
    
    au1v1 = df.inner(sigma1(u1), epsilon(v1))*(dx(omega1) + dx(interf))  + Gh1 \
        + gamma_p*h**(-2)*df.inner(u1,v1)*dx(interf) \
        + gamma_u*df.inner(sigma1(u1), sigma1(v1))*dx(interf) 
    au1z1 = gamma_u*df.inner(sigma1(u1), z1)*dx(interf)
    au1v2 = -gamma_p*h**(-2)*df.inner(u1,v2)*dx(interf)
    au1q = gamma_p*h**(-3)*df.inner(u1,q*phi)*dx(interf)
    
    au2v1 = -gamma_p*h**(-2)*df.inner(u2,v1)*dx(interf)
    au2v2 = df.inner(sigma2(u2), epsilon(v2))*(dx(omega2) + dx(interf)) + Gh2 \
        + gamma_p*h**(-2)*df.inner(u2,v2)*dx(interf) \
        + gamma_u*df.inner(sigma2(u2), sigma2(v2))*dx(interf)  
    au2z2 = gamma_u*df.inner(sigma2(u2), z2)*dx(interf)
    au2q = -gamma_p*h**(-3)*df.inner(u2,q*phi)*dx(interf)
    
    ay1v1 = (df.inner(df.dot(y1("+"),n("+")),v1("+")))*dS(gamma1) + gamma_u*df.inner(y1,sigma1(v1))*dx(interf)
    ay1z1 = gamma_div*df.inner(df.div(y1),df.div(z1))*dx(interf) + gamma_u*df.inner(y1,z1)*dx(interf) \
            + gamma_y*h**(-2)*df.inner(df.dot(y1,df.grad(phi)),df.dot(z1,df.grad(phi)))*dx(interf)
    ay1z2 = - gamma_y*h**(-2)*df.inner(df.dot(y1,df.grad(phi)),df.dot(z2,df.grad(phi)))*dx(interf)

    ay2v2 = (df.inner(df.dot(y2("+"),n("+")),v2("+")))*dS(gamma2) + gamma_u*df.inner(y2,sigma2(v2))*dx(interf)
    ay2z1 = -gamma_y*h**(-2)*df.inner(df.dot(y2,df.grad(phi)),df.dot(z1,df.grad(phi)))*dx(interf)  
    ay2z2 = gamma_div*df.inner(df.div(y2),df.div(z2))*dx(interf) \
            + gamma_u*df.inner(y2,z2)*dx(interf) \
            + gamma_y*h**(-2)*df.inner(df.dot(y2,df.grad(phi)),df.dot(z2,df.grad(phi)))*dx(interf)

    apv1 =  gamma_p*h**(-3)*df.inner(p*phi,v1)*dx(interf)
    apv2 = -gamma_p*h**(-3)*df.inner(p*phi,v2)*dx(interf)
    apq  = gamma_p*h**(-4)*df.inner(p*phi,q*phi)*dx(interf)
    
    lv1 = df.dot(f,v1)*(dx(omega1) + dx(interf)) 
    lv2 = df.dot(f,v2)*(dx(omega2) + dx(interf)) 
    lz1 = gamma_div * df.inner(f,df.div(z1))*dx(interf) 
    lz2 = gamma_div * df.inner(f,df.div(z2))*dx(interf) 

    A = [[au1v1, au1v2, au1z1, 0.0, au1q], 
            [au2v1, au2v2, 0.0, au2z2, au2q],
            [ay1v1, 0.0, ay1z1, ay1z2, 0.0],
            [0.0, ay2v2, ay2z1, ay2z2, 0.0],
            [apv1,  apv2, 0.0, 0.0,  apq]]
    L = [lv1, lv2, lz1, lz2, 0.0]

    AA = mph.block_assemble(A)
    LL = mph.block_assemble(L)
    # definition of the Dirichlet conditions (top, bottom, left and right sides of the square)
    def boundary(x, on_boundary):
        return on_boundary and (df.near(x[0], 0.0) or df.near(x[0], 1.0) or df.near(x[1],1.0) or df.near(x[1],0.0))
    bc1 = mph.DirichletBC(W.sub(0), u_D, boundary) # apply DirichletBC on Omega1
    bc2 = mph.DirichletBC(W.sub(1), u_D, boundary) # apply DirichletBC on Omega2
    bcs = mph.BlockDirichletBC([bc1,bc2])
    bcs.apply(AA)
    bcs.apply(LL)
    UU = mph.BlockFunction(W)
    mph.block_solve(AA, UU.block_vector(), LL)

    # Solution on Omega1
    u_h1 = UU[0] #df.project(UU[0], V)
    # Solution on Omega2
    u_h2 = UU[1] #df.project(UU[1], V)
    # Solution on Omega 
    
    u_h = b*UU[0] + (1.0-b)*UU[1] 
    u_h = df.project(u_h, V)
    
    error_l2.append((df.assemble((((u_ex-u_h))**2)*df.dx)**(0.5))/(df.assemble((((u_ex))**2)*df.dx)**(0.5)))            
    error_h1.append((df.assemble(((df.grad(u_ex-u_h))**2)*df.dx)**(0.5))/(df.assemble(((df.grad(u_ex))**2)*df.dx)**(0.5)))  
    
# plot error on the figure matplotlib   
plt.loglog(hh,error_h1,'o--', label=r'$\phi$-FEM $H^1$')
plt.loglog(hh,error_l2,'o-', label=r'$\phi$-FEM $L^2$')
plt.xlabel("$h$")
plt.ylabel(r'$\frac{\|u-u_h\|}{\|u\|}$')
plt.legend(loc='lower right', ncol=2)
plt.title(r'Relative error : $ \frac{\|u-u_h\|}{\|u\|} $ for $L^2$ and $H^1$ norms', y=1.025)
plt.tight_layout()
plt.show()

#  Write the output file for latex
if E1 == E2 and nu1 == nu2:
    f = open('first_scheme_elasticity_same_parameters_P{name0}.txt'.format(name0=degV),'w')
else :
    f = open('first_scheme_elasticity_different_parameters_P{name0}.txt'.format(name0=degV),'w')
    
f.write('(E_1, nu_1, lambda_1, mu_1) = ( ' + str(E1) + ', ' + str(nu1) + ', ' + str(lambda_1) + ', ' + str(mu1) + ') \n')  	
f.write('(E_2, nu_2, lambda_2, mu_2) = ( ' + str(E2) + ', ' + str(nu2) + ', ' + str(lambda_2) + ', ' + str(mu2) + ') \n')
f.write('relative L2 norm phi fem: \n')	
output_latex(f, hh, error_l2)
f.write('relative H1 norm phi fem : \n')	
output_latex(f, hh, error_h1)
f.close()
