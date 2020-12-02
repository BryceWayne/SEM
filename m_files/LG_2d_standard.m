% Solve Poisson eqn on [-1,1]x[-1,1] with u=0 on boundary
% D= differentiation matrix -- Chebyshev Diff

clear all
N = 31;


% For Cheb Diff
% [D,x] = cheb(N);

% For Legendre Diff
x = legslbndm(N+1);
D = legslbdiff(N+1,x);

y = x;


% Set up grids and tensor product Laplacian, and solve for u:
[x1,y1] = meshgrid(x,y);
[xx,yy] = meshgrid(x(2:N),y(2:N));

m = rand(2,1);
w = randn(4,1)*pi/2;

% External forcing term
f_full = m(1)*cos(w(1)*x1 + w(2)*y1) + m(2)*sin(w(3)*x1 + w(4)*y1);%-0.7*sin(3*x1) + 0.3*cos(4*y1) + exp(x1 + 2*y1);


% stretch 2D grids to 1D vectors
xx = xx(:); yy = yy(:);
% source term function
f = m(1)*cos(w(1)*xx + w(2)*yy) + m(2)*sin(w(3)*xx + w(4)*yy);%-0.7*sin(3*xx) + 0.3*cos(4*yy) + exp(xx + 2*yy);

D2 = D^2; D2 = D2(2:N,2:N); I = eye(N-1); % Laplacian
L = kron(I,D2) + kron(D2,I);
u = -L\f;

% Reshape long 1D results onto 2D grid:
uu = zeros(N+1,N+1);
uu(2:N,2:N) = reshape(u,N-1,N-1);
[xx,yy] = meshgrid(x,y);
figure(1), clf, mesh(xx,yy,uu)




m=2;
for l=0:m
    for j=0:m
        
        phi1 = lepoly(l,x) - lepoly(l+2,x);
        phi1_x = lepolyx(l,x) - lepolyx(l+2,x);
        
        phi2 = lepoly(j,x) - lepoly(j+2,x);
        phi2_x = lepolyx(j,x) - lepolyx(j+2,x);
        
        
        for i=1:N+1
            u_x(i,:) = D*uu(i,:)';
            u_y(:,i) = D*uu(:,i);
            
            ux(i) = sum(u_x(i,:)'.*(phi1_x)*2/(N*(N+1))./(lepoly(N,x).^2));
            uy(i) = sum(u_y(:,i).*(phi2_x)*2/(N*(N+1))./(lepoly(N,x).^2));
            
            fx(i) = sum(f_full(i,:)'.*(phi1)*2/(N*(N+1))./(lepoly(N,x).^2));
        end
        
        
        lhs((m+1)*l+j+1) = sum(ux'.*phi2*2/(N*(N+1))./(lepoly(N,x).^2)) + sum(uy'.*phi1*2/(N*(N+1))./(lepoly(N,x).^2));
        rhs((m+1)*l+j+1) = sum(fx'.*phi2*2/(N*(N+1))./(lepoly(N,x).^2));
        
    end
end

err = lhs - rhs;
lhs
rhs
err







% Interpolate to finer grid and plot:
[xxx,yyy] = meshgrid(-1:.04:1,-1:.04:1);
uuu = interp2(xx,yy,uu,xxx,yyy,'spline');
figure(2), clf, mesh(xxx,yyy,uuu), colormap(1e-6*[1 1 1]);
xlabel x, ylabel y, zlabel u