clear all;
close all;
format short e

%
% Main file of Legendre Gelerkin method
% Two more files are needed here: lepoly.m and legslbndm.m



N = 3;
ep=10^-2;
sigma = 1;
% Compute LGL points
x = legslbndm(N+1);
D = legslbdiff(N+1,x);
% Given Boundary conditions, we can find the coefficient of Legendre
% polynomials
a(1:N+1) = 0;
b(1:N+1) = -1;



% External force
% f(1:N+1) = sin(2*pi*x) + 4*pi*cos(2*pi*x) + 2*pi*cos(2*pi*x).*(x - 1) - 4*pi^2*sin(2*pi*x).*(x - 1);
%f(1:N+1) = sin(x).^2;


f(1:N+1) = 1/2;


% f(1:N+1) = 1/2;
% u_ex = sin(2*pi*x).*(1-x);
phi = get_phi(N,x,sigma,ep);
res = -(exp(-(sigma + 1)/ep) - 1)/(sigma + 1);

% plot(x,phi,'-ok','linewidth',1.5'linewidth',1.5)
% title('Boundary layer element')
% xlabel('X-axis')
% return


S = zeros(N-1,N-1);


% Assign the linear operator
for ii=1:N-1
    k=ii-1;
    %s_diag(ii) = -(4*k+6)*b(ii);
    
    phi_k_a12 = (lepoly(k,x) + a(ii)*lepoly(k+1,x) + b(ii)*lepoly(k+2,x));
    phi_k_M = D*phi_k_a12;
    phi_k_s = D*phi_k_M;
    
    for jj=1:N-1
        if abs(ii-jj) <= 2
            l = jj-1;
            psi_l_M = (lepoly(l,x) + a(jj)*lepoly(l+1,x) + b(jj)*lepoly(l+2,x));
            M(jj,ii) = sum(psi_l_M.*phi_k_M*2/(N*(N+1))./(lepoly(N,x).^2));
        end
        
        if ii<=jj-2
            q = jj-1;
            phi_q = lepoly(q,x) + a(jj)*lepoly(q+1,x) + b(jj)*lepoly(q+2,x);
            S(jj,ii) = sum(phi_k_s.*phi_q*2/(N*(N+1))./(lepoly(N,x).^2));
        end
    end
    
    a_12(ii) = sum(res.*phi_k_a12*2/(N*(N+1))./(lepoly(N,x).^2));
    a_21(ii) = sum( (   -ep*phi_k_s - phi_k_M   ).*phi'*2/(N*(N+1))./(lepoly(N,x).^2));
end

a_22 = sum(res.*phi.'*2/(N*(N+1))./(lepoly(N,x).^2));


Mass_mat = -ep*S-M;
Mass_mat(1:N-1,N) = a_12;
Mass_mat(N,1:N-1) = a_21;
Mass_mat(N,N) = a_22;
Mass = sparse(Mass_mat);




% Backward Legendre Transform32
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


bar_f_end = 0;
for ii=1:N+1
    k=ii-1;
    bar_f_end = bar_f_end + g(ii)*2/(N*(N+1))*sum(phi'.*(lepoly(k,x))./(lepoly(N,x).^2));
end

bar_f(N) = bar_f_end;



% Main solver
u_temp = Mass\bar_f';
u_sol = zeros(N+1,1);



for ij=1:N-1
    i_ind = ij-1;
    u_sol = u_sol + u_temp(ij)*(  lepoly(i_ind,x) + a(ij)*lepoly(i_ind+1,x) + b(ij)*lepoly(i_ind+2,x)  );
end
u = u_sol + u_temp(N)*phi';





plot(x,u,'-xk','linewidth',1.5)
title('Numerical solution','Interpreter','latex','FontSize',15)
xlabel('X-axis')
ylabel('Y-axis')


exact = ((exp(-(x+1)/ep) - 1)/(exp(-2/ep)-1) - (x+1)/2);

hold on

plot(x,exact,'-r','linewidth',1.5)

legend('Exact solution','Numerical solution')
norm(u-exact,2)/norm(exact,2)

