function z=legslbndm(n)

%z=leglndm(n) returns n Legendre-Gauss-Lobatto quadrature nodes with
%   z(1)=-1, z(n)=1 computed by the eigen-method
%   Recall that the interior nodes are zeros of L_{N-1}'(x)=c J_{n-2}^{1,1}(x) 
%   See Page 84 of the book: J. Shen, T. Tang and L. Wang, Spectral Methods:
%   Algorithms, Analysis and Applications, Springer Series in Compuational
%   Mathematics, 41, Springer, 2011.     
%
%Last modified on August 30, 2011       

if n<=1, disp('n should be bigger than 1'); z=[]; return; end;
if n==2, z=[-1;1]; return; end;
if n==3, z=[-1;0;1]; return; end;

av=zeros(1,n-2); 
j=[1:n-3]'; bv=j.*(j+2)./((2*j+1).*(2*j+3));
A=diag(av)+diag(sqrt(bv),1)+diag(sqrt(bv),-1);  % form the Jacobi matrix (3.142) with alpha=beta=1
z=sort(eig(sparse(A)));     % find the eigenvalues

z=[-1;z;1];
 return