function [A, S] = modelMatrix(s, m, N)

%% figure out how to set n properly w.r.t. stride
n = N - (m - 1);

block = zeros(m-1, m);
block(:, 1) = -1;
block(:, 2:end) = eye(m-1, m-1);

h = (m-1) - 1;
w = m - 1;
S = sparse(3*n, n);
for t=0:(n-1)

    i = ((m-1)*t) + 1;
    j = (s*t) + 1;

    S(i:i+h, j:j+w) = block;

end

A = kron(S, eye(3));

end
