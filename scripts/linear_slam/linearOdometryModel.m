function A = linearOdometryModel(numPoints, conn, stride)

connBlock = zeros(conn-1, conn);
connBlock(:, 1) = -1;
connBlock(:, 2:end) = eye(conn-1);

numBlocks = floor(numPoints / stride);

modelHeight = numBlocks * (conn-1);
modelWidth = numPoints;
A = sparse(modelHeight, modelWidth);

dh = (conn-1) - 1;
dw = conn - 1;
for b=0:(numBlocks-1)
    
    i = ((conn-1) * b) + 1;
    j = (b * (stride)) + 1;
    
    A(i:i+dh, j:j+dw) = connBlock;

end
A = A(:, 1:numPoints);

% A = kron(A, eye(3));

