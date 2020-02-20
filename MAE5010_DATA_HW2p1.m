% Code by Christopher E. Petrin
% Prepared for MAE 5010: Data Assimilation (Spring 2020, Dr. Omer San)
% For Homework #2, Problem #1

%Problem description: Consider a scalar dynamics:
%	x_k+1 = a*x_k and z_k = x_k + v_k where v_k ~ N(0,var), var = 0.5

%% STEP A: 
%  Pick a = 1 and x0 = 1, and generate the states {x_k} from k = 0 k = N,
%   with N = 50.
for i = 1:100 % Run code 100 times to get statistics on each method.
clearvars -except i xf %Get rid of whatever is in the workspace.
%INPUTS

a = 1; %In notation, this is M.
	M = a;
H = 1;
x0 = 1;
var = 0.5;
	R = var;
N = 50;
conCrit = -2;

%Initialize x(), z() & compute v(k)
x = zeros(N,1);
z = zeros(N,1);

v = 0 + (0.5 - 0) * randn(N,1);

%% STEP B & C: 
%	 B: Generate z_k for k = 1 to N and store this sequence
%  C: Define observation sets as follows:
%		  O1 = {z1,z2,z3,...,z50}
%			O2 = {z1,z5,z10,...,z50}
%			O3 = {z1,z10,z20,...,z50}
x(1) = x0;
for k = 1:N
	x(k+1) = a*x(k);
	z(k) =	H*x(k) + v(k);
	
	if k == 1
		O1(k) = z(k);
		O2(k) = z(k);
		O3(k) = z(k);
	elseif mod(k, 5) == 0
		O1(size(O1,1)+1,1) = z(k);
		O2(size(O2,1)+1,1) = z(k);
		
		if mod(k,10) == 0
			O3(size(O3,1)+1,1) = z(k);
		end
	else
		O1(size(O1,1)+1,1) = z(k);
	end
end

%% STEP D:
%	 Start the model from x0 = 0.5 and use each of the observation sets and
%	 estimate the optimal x0 using the gradient method.
%	 Run for i = 1:3 for O1, O2, O3.
for j = 1:3
	if j == 1
		O = O1;
	elseif j == 2
		O = O2;
	elseif j == 3
		O = O3;
	end
	N = size(O,1);
	convTest = 0;
	x0 = 0.5;
	
	while convTest == 0	%Start of gradient loop
		% Before minimization, must run optimization loop:
		% Optimization - 4D Var I
		%		Step 1) Compute {x(k)} from k = 1 to N using x_k+1 = M*x_k
		%		AND
		%		Step 2) Given observations {Z(k)} from k = 1 to N, 
		%			compute f(k) = H'*inv( R(k) )*( z(k) - H*x(k) )
		%			Note: for z(k) in the above eqn, must use O1(k), O2(k), and O3(k)
		x4D(:,1) = x0;
		for k = 1:N
			x4D(:,k+1) = M*x4D(:,k);
			f(k) = H'*pinv(R)*(O(k) - H*x4D(:,k));
		end
		%		Step 3) Set l(N) = f(N) and solve l(k) = M'*l(k+1) to find l(k)
		%		AND
		%		Step 4) Compute Grad(x(0))J(x(0)) = -M'*l(1)
		l(N) = f(N);
		for k = N-1:-1:1
			l(k) = M'*l(k+1) + f(k);
			GradJ = -M'*l(1);
		end

		% Minimization - Steepest Descent/Gradient
		%		Step 0) Residual r = B - A*x0 
		%			In this case is r = -M'*l(1) = GradJ, and A = -M', and B = 0.
		r = -GradJ;
		A = M';
		%		Step 1) Find step length: alpha = ( r' * r )/( r' * A * r )
		alpha = 0.1;
		%		Step 2) Compute new optimal x

		%		Step 3) Test for convergence
		testCrit = abs(-M'*l(1));
		if testCrit <= 1*10^conCrit
			convTest = 1;
		end

		% Set an adaptive learning step for convergence. Basically, step is
		%  always 3 orders of magnitude lower than the testCriteria rounded up
		%  to the nearest power of 10.
		alpha = 10^(ceil(log10(testCrit))-3);

		x0 = x0 - alpha*GradJ/norm(GradJ);
		clear x4D f l GradJ alpha x0new r A
	end
	
	xf(i,j) = x0;
end 
end

% Collect statistics for each method. Postscrips are as follows:
%		m = mean of all 100 runs
%		s = stdev of all 100 runs
%		p = precision
%		a = accuracy (real should be x0Real + cov/2)
Real = 1;
O1m = mean(xf(:,1));
O1s = std(xf(:,1));
O1p = O1s/O1m * 100;
O1a = O1m/Real*100;

O2m = mean(xf(:,2));
O2s = std(xf(:,2));
O2p = O2s/O2m * 100;
O2a = O2m/Real*100;

O3m = mean(xf(:,3));
O3s = std(xf(:,3));
O3p = O3s/O3m * 100;
O3a = O3m/Real*100;