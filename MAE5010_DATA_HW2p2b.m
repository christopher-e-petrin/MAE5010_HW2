% Code by Christopher E. Petrin
% Prepared for MAE 5010: Data Assimilation (Spring 2020, Dr. Omer San)
% For Homework #2, Problem #2

%Problem description: Consider a scalar dynamics:
%	x_k+1 = 4*x_k(1-x_k) and z_k = x_k + v_k where v_k ~ N(0,var), var = 0.5

%% STEP A: 
%  Pick a = 1 and x0 = 1, and generate the states {x_k} from k = 0 k = N,
%   with N = 50.

clearvars %Get rid of whatever is in the workspace.
%SETUP)
%	INPUT
x0R = 0.5;	%x0 used to generate observations
x0 = 0.8;	%Initial x0 guess
N = 20;
var = 0.5;
H = 1;
tic
%	Calculate variance matrix
v = 0 + (var - 0) * rand(N,1);
R = 0 + (var - 0) * rand(N,1);
%	Create symbolic M and dM, with y as a symbolic variable placeholder for
%	 x(k)
syms M(y) dM(y)
M(y) = 4*y*(1-y);
dM(y) = diff(M,y);

%Step 1) Set x0 = 0.5 (done above), and generate observation as 
%	z(k) = x(k) + v(k) for k = 1:20, where v(k) ~ N(0,var) and var = 0.5.
x(1) = x0R;
for k = 1:N
	x(k+1) = M(x(k));
	z(k) = x(k)+v(k);
end
%Step 2) Start the model from x0 = 0.8 (above) and assimilate all 20
%	observations to estimate the optimal x0.
convTest = 0;
iter = 1;

%	Start outside loop for gradient minimization
for j = 1:N
%	Before minimization, must run optimization loop:
%	Optimization: 4D-VAR III (nonlinear M, linear H)
%	Step 1) Compute x(k) from k = 1:N, using x(k+1) = M(x(k))
%	AND
%	Step 2) Compute Dx(M) and evaluate Dxk(M) for k = 0:N-1
% AND
%	Step 3) Compute f(k) = D(k)'*pinv(R(k))*(Z(k) - h(k))
	x(1) = x0;
	for k = 1:N
		x(k+1) = M(x(k));
		%Note computed function for Dx(M) above as dM(y)
		D(k) = double(dM(x(k)));
		f(k) = D(k)'*pinv(R(k))*(z(k) - H*x(k));
	end
%	Step 4) Solve l(k) = D(k)' * M * l(k+1), k = N-1:-1:N
	l(N) = f(N);
	for k = N-1:-1:1
		l(k) = D(k)' * l(k+1) + f(k);
	end
% Step 5) GradJ = -D(0)' * l(1)
	GradJ = -D(1)'*l(1);
	
% Minimization algorithm: Conjugate Gradient
%	Step 0) set r & p
	r = GradJ;
	p(1) = r;
	for i = 1:N
%	Step 1) Calculate alpha
%		To do this, following the method in the notes:
%			First, set g(alpha) = M(x(k+1)) = M( x(k) + alpha*p )
%			Then, find derivative of g with respect to alpha (dg/da)
%			Set dg/da equal to 0, solve for alpha.
%		This method yields an alpha equation of (4p - 8p*x)/8p^2
%			which can be simplified to p*dM(x)/8p^2.
%		Note: this could also be solved symbolically in MATLAB as follows:
%			syms g(a)
%			g(a) = M(y + r*a);
%			dg(a) = diff(g,a);
%			eqn = dg(a) == 0;
%			S(y) = solve(eqn,a);
%			alpha = double(S( x(1) ));
		alpha = p(i)*double(dM(x(1)))/(8*p(i)^2);
		r(i+1) = r(i) - alpha*(-4)*p(i);
		testCrit = r(i+1)' * r(i+1);
		if testCrit <= 0.001
			convTest = 1;
		end
		beta = (r(i+1)'*r(i+1))/(r(i)'*r(i));
		p(i+1) = r(i+1) + beta*p(i);
		% Create an adaptive factor which helps create more stable convergence
		fact = 0.1*abs(GradJ)/abs(GradJ);
		x0 = x0 +  fact*alpha*r(i);
	end
end
t = toc
x0
abs(x0-x0R)/x0R*100
