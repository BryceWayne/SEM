clear all;
close all;
format short e

%     
% Main file of Legendre Gelerkin method
% Classical method
% Two more files are needed here: lepoly.m and legslbndm.m
% u_t -\ep u_xx + u*u_x = f
% Backward Euler scheme


N = 32;
ep= 0.1;
tol = 1e-9;
dt = 10^-4;
% Compute LGL points
x = legslbndm(N+1);
D = legslbdiff(N+1,x);
T = 1;
% Given Boundary conditions, we can find the coefficient of Legendre
% polynomials
a(1:N+1) = 0;
b(1:N+1) = -1;


% Random force
m1 = normrnd(0,1);
m2 = normrnd(0,1);
w1 = normrnd(0,1);
w2 = normrnd(0,1);
force = m1*sin(2*pi*x*w1)' + m2*cos(2*pi*x*w2)';






for ii=1:N-1
    k=ii-1;
    s_diag(ii) = -(4*k+6)*b(ii);
    
    phi_k_M = (lepoly(k,x) + a(ii)*lepoly(k+1,x) + b(ii)*lepoly(k+2,x));
    
    for jj=1:N-1
        if abs(ii-jj) <= 2
            l = jj-1;
            psi_l_M = (lepoly(l,x) + a(jj)*lepoly(l+1,x) + b(jj)*lepoly(l+2,x));
            M(jj,ii) = sum(psi_l_M.*phi_k_M*2/(N*(N+1))./(lepoly(N,x).^2));
        end
    end

end

S = diag(s_diag);
Mass = sparse(ep*S + 1/dt*M);

% Test exact solution
% cos(t)sin(pi*x)
% Initial condition
u_pre = sin(pi*x);
t_f = T/dt;

for t_idx=1:t_f
    
    u_old = x'*0;
    force = force*cos(t_idx*dt);
    
    % For exact solution example
    %force = sin(pi*x).*(pi^2*cos(t_idx*dt) - sin(t_idx*dt) + pi*cos(pi*x).*cos(t_idx*dt)^2);
    %force = force.';
    
    err = 1;
    
    while err > tol
        f = force - u_old.*((D*u_old')') + 1/dt*u_pre.';
        
        
        % Backward Legendre Transform
        for i=1:N
            k=i-1;
            g(i) = (2*k+1)/(N*(N+1))*sum(f'.*(lepoly(k,x))./(lepoly(N,x).^2));
        end
        g(N+1) = 1/(N+1)*sum(f'./lepoly(N,x));
        
        
        % Evaluate rhs of the weak form
        for i=1:N-1
            k=i-1;
            bar_f(i) = g(i)/(k+1/2) + a(i)*g(i+1)/(k+3/2) + b(i)*g(i+2)/(k+5/2);
        end
        
        
        % Main solver
        u = Mass\bar_f';
        
        u_sol = zeros(N+1,1);
        
        for ij=1:N-1
            i_ind = ij-1;
            u_sol = u_sol + u(ij)*(  lepoly(i_ind,x) + a(ij)*lepoly(i_ind+1,x) + b(ij)*lepoly(i_ind+2,x)  );
        end
        
        err = max(u_sol - u_old');
        u_old = u_sol';
        
    end
    
    v(t_idx,:) = u_sol;
    u_pre = u_sol;
    
end


%norm(v(end,:) - cos(1)*sin(pi*x)')
surf(v); 
shading interp

return
























%%%%%%
%%% Not use the below for the moment
%%%%

cumulative_error = 0;


for l=0:15
    temp = 0;
    diffusion = -ep*(4*l+6)*(-1)*u(l+1);

    
    temp = 1/2*(u_sol.^2).*(lepolyx(l,x) - lepolyx(l+2,x));

    convection = sum(temp*2/(N*(N+1))./(lepoly(N,x).^2));
    rhs = sum(force'.*(lepoly(l,x) - lepoly(l+2,x))*2/(N*(N+1))./(lepoly(N,x).^2));
    cumulative_error = cumulative_error + abs(diffusion - convection - rhs);
end

cumulative_error

plot(x,u_sol,'-ok','linewidth',1.5)
title('Numerical solution')
xlabel('X-axis')

%DE = -ep*D^2*u_sol + D*u_sol.*u_sol - force'
