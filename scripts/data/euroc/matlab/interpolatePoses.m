function Q = interpolatePoses(P, t, tq)

% to Lie algebra
S = zeros(size(P));
nPoses = size(P, 3);
for i=1:nPoses
    S(:, :, i) = logm(P(:, :, i));
end

% perform interpolation
Sq = spline(t, S, tq);
% Sq = interp1(t, S, tq, 'cubic');

% return
Q = zeros(size(Sq));
nPoses = size(Sq, 3);
for i=1:nPoses
    Q(:, :, i) = expm(Sq(:, :, i));
end

end
